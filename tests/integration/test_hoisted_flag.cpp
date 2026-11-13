// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// Regression guard for the hoisted-flag deadlock class.
//
// The persistent worker waits on cross-worker flags. If a flag is a plain
// (non-atomic, non-volatile) word, the compiler may prove it cannot change
// inside the loop, load it once, and spin forever. The guard is the deadline:
// the flag is set by a *different* worker after a delay, so a correct
// re-reading loop observes the set value and finishes before the deadline,
// while a loop that cached the initial value exhausts the deadline and fails.
//
// The test exercises the project's real wait primitives:
//   - a device-scope atomic load (the `stopReason` / quiescence word class),
//     the CUDA variant, where the flag is raised by another block;
//   - a host-scope atomic load (the `initDone` / host-control word class), the
//     CPU variant, where the flag is raised by another thread.
//
// The flag is deliberately never cached before the loop: the test asserts the
// re-read, not the source. A correct build finishes in well under the budget;
// a hoisting compiler exhausts it and the test fails.

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/device_atomic.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#if GRAPH_HAS_CUDA
#    include <cuda_runtime.h>
#endif

namespace
{
    /** How long a worker may spin before it reports the flag never arrived.
     *
     * A correct re-reading loop finishes in a few hundred microseconds (the
     * raise delay dominates); the budget is orders of magnitude above that, so
     * a hoisted load (which never sees the set value) exhausts it while a
     * correct loop never comes close. */
    inline constexpr std::uint64_t spinBudget = 10'000'000u;
    /** The delay before the other worker raises the flag.
     *
     * Nonzero so the first observations are genuinely the unset value: the test
     * proves the wait *observes a later write*, not merely that a flag already
     * set at spin start is visible. */
    inline constexpr std::uint64_t raiseDelayMicroseconds = 300u;

    /** Spin on a device-scope atomic load of @p flag until it is nonzero or the
     *  budget is exhausted.
     *
     * The acquire load is the only way out of the loop. `memory_order_acquire`
     * pairs with the writer's release store and forces a re-read on every
     * iteration; a plain member would let the compiler cache the value and
     * never observe the writer. The pause gives the writer a real chance to land
     * the store without turning the wait into a full-speed spin that would burn
     * the budget in well under a second. Returns the number of iterations
     * consumed, which is what a hoisting loop reports (the whole budget). */
    template<typename T_Acc>
    ALPAKA_FN_ACC std::uint64_t spinOnDeviceFlag(T_Acc const& acc, std::uint64_t* flag) noexcept
    {
        std::uint64_t iterations = 0u;
        while(graph::detail::loadDeviceAcquire(acc, flag) == 0u)
        {
            if(iterations >= spinBudget)
                break;
            ++iterations;
#if defined(__CUDA_ARCH__)
            __nanosleep(256u);
#endif
        }
        return iterations;
    }

    /** The writer: wait the delay, then raise @p flag with a release store. */
    template<typename T_Acc>
    ALPAKA_FN_ACC void raiseDeviceFlag(T_Acc const& acc, std::uint64_t* flag) noexcept
    {
#if defined(__CUDA_ARCH__)
        __nanosleep(static_cast<unsigned>(raiseDelayMicroseconds * 1000u));
#else
        (void) acc;
#endif
        graph::detail::storeDeviceRelease(acc, flag, std::uint64_t{1u});
    }

    /** One worker. Block 0 raises the flag on the delay; every other block
     *  spins on it with the budget and records the iterations it consumed.
     *
     * The raise and the spin run in *separate blocks* (block 0 writes, block 1
     * reads), which is the cross-worker case a plain member cannot survive. The
     * two blocks are a trivial single-wave launch on any device, so no block
     * misses the deadline because it is not scheduled: the deadline bounds the
     * wait, not the launch. The reader stores its result for the host to read
     * back, because a kernel cannot return a value. */
    struct HoistedFlagKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, std::uint64_t* state) const
        {
            auto const blockIndex = static_cast<std::uint32_t>(
                acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks)[0]);
            if(blockIndex == 0u)
            {
                raiseDeviceFlag(acc, state);
                return;
            }
            std::uint64_t const iterations = spinOnDeviceFlag(acc, state);
            // One slot per reader block; block 1 is the only reader.
            state[1u + blockIndex] = iterations;
        }
    };

    /** CPU / host variant: the writer is a host thread, the reader is the
     *  calling thread. The flag is a `std::atomic` (the host-control word
     *  class, `initDone`). The writer sleeps the delay then raises it with a
     *  release store; the reader spins on an acquire load with the budget. A
     *  host build that cached the flag in a register would burn the whole
     *  budget and fail. Returns the iterations consumed. */
    std::uint64_t spinOnHostFlag(std::atomic<std::uint64_t> const& flag)
    {
        std::uint64_t iterations = 0u;
        while(flag.load(std::memory_order_acquire) == 0u)
        {
            if(iterations >= spinBudget)
                break;
            ++iterations;
            std::this_thread::yield();
        }
        return iterations;
    }

    std::uint64_t runHostPair()
    {
        std::atomic<std::uint64_t> flag{0u};
        std::thread writer(
            [&flag]()
            {
                std::this_thread::sleep_for(std::chrono::microseconds{raiseDelayMicroseconds});
                flag.store(1u, std::memory_order_release);
            });
        std::uint64_t const iterations = spinOnHostFlag(flag);
        writer.join();
        return iterations;
    }
} // namespace

#if GRAPH_HAS_CUDA
TEST_CASE("device flag set by another block is observed before the budget", "[graph][pool][hoisted][gpu]")
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.isAvailable());
    REQUIRE(selector.getDeviceCount() > 0u);
    auto device = selector.makeDevice(0);
    auto executor = alpaka::onHost::defaultExecutor(device);

    // Slot 0 is the flag; slot 2 holds block 1's reported iteration count.
    auto buffer = alpaka::onHost::alloc<std::uint64_t>(device, 4u);
    auto hostBuffer = alpaka::onHost::allocHost<std::uint64_t>(4u);
    auto queue = device.makeQueue();
    for(std::uint32_t i = 0u; i < 4u; ++i)
        alpaka::onHost::data(hostBuffer)[i] = 0u;
    alpaka::onHost::memcpy(queue, buffer, hostBuffer);
    alpaka::onHost::wait(queue);

    // Two blocks: block 0 writes, block 1 reads. One trivial wave, so both are
    // resident and the budget is a real bound on the wait, not the launch.
    auto const spec = alpaka::onHost::ThreadSpec{2u, 32u, executor};
    queue.enqueue(spec, HoistedFlagKernel{}, alpaka::onHost::data(buffer));
    alpaka::onHost::wait(queue);

    alpaka::onHost::memcpy(queue, hostBuffer, buffer);
    alpaka::onHost::wait(queue);

    // The writer must have raised the flag.
    REQUIRE(alpaka::onHost::data(hostBuffer)[0u] == 1u);
    // The reader must have observed it: a hoisting reader burns the whole
    // budget; a correct reader stops as soon as it sees the set value, far
    // below the budget.
    INFO(
        "device reader consumed " << alpaka::onHost::data(hostBuffer)[2u] << " of " << spinBudget
                                  << " budget iterations");
    REQUIRE(alpaka::onHost::data(hostBuffer)[2u] < spinBudget);
}
#endif // GRAPH_HAS_CUDA

TEST_CASE("host flag set by another thread is observed before the budget", "[graph][pool][hoisted][cpu]")
{
    std::uint64_t const iterations = runHostPair();
    INFO("host reader consumed " << iterations << " of " << spinBudget << " budget iterations");
    REQUIRE(iterations < spinBudget);
}
