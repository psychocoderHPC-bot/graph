// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// Unit tests of the barrier divergence diagnostic
// (`graph/detail/barrier_diagnostics.hpp`). The diagnostic turns a divergent
// arrival at a warp barrier (which with one warp per block would deadlock the
// block barrier) into a recorded call site. Two tests:
//
// 1. A converged full-width barrier with the diagnostics enabled records
//    zero divergences (the clean path the pool runs).
// 2. A deliberately divergent arrival (only a subset of the lanes calls the
//    barrier) records the divergence with the expected barrier id, worker,
//    and stage, and skips the barrier when `skipOnDivergence` is set so the
//    test fails loudly instead of hanging.
//
// Test 2 is CUDA-only: the CPU backend has no warps, so there is no
// divergence to observe. On CUDA the kernel is launched with one block of 32
// threads (one warp) and a lane guard that leaves a subset active; the
// diagnostic's own first-active-lane logic picks the lowest active lane index
// as the recording lane.

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/barrier_diagnostics.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>

#include <cstdint>

#if GRAPH_HAS_CUDA

namespace
{
    auto makeCudaDevice()
    {
        auto const spec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
        auto selector = alpaka::onHost::makeDeviceSelector(spec);
        REQUIRE(selector.isAvailable());
        REQUIRE(selector.getDeviceCount() > 0u);
        return selector.makeDevice(0);
    }

    /** Kernel: every lane of the block calls the barrier. The barrier must
     *  observe the full active mask and record no divergence.
     *
     * Results:
     *   [0] = divergences recorded (expected 0)
     *   [1] = firstBarrierId (expected 0xFFFFFFFF, the "no record" sentinel)
     */
    struct ConvergedKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, std::uint32_t barrierId, void* state, void* results) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            auto* diag = static_cast<graph::BarrierDiagnosticsState*>(state);
            auto* out = static_cast<std::uint32_t*>(results);
            graph::WarpGroup<Acc> warp{acc};
            graph::BarrierDiagnosticsConfig cfg;
            cfg.enabled = true;
            cfg.skipOnDivergence = false;
            auto diagnostics = graph::WarpBarrierDiagnostics<Acc>::make(diag, cfg);
            graph::WarpBarrier<Acc> barrier{diagnostics};
            constexpr std::uint32_t blockThreads = 32u;
            (void) barrier.barrier(acc, blockThreads, warp, barrierId, 0u, 0u);
            if(warp.isLeader())
            {
                out[0] = static_cast<std::uint32_t>(diag->divergenceCount);
                out[1] = diag->firstBarrierId;
            }
        }
    };

    /** Kernel: only the lanes with index below `activeCount` call the barrier.
     *  The diagnostic must record the divergence (the barrier is skipped so
     *  the kernel finishes), and the first record must name the call site.
     *
     * Results:
     *   [0] = divergences recorded (expected >= 1)
     *   [1] = firstBarrierId (expected the call-site id)
     *   [2] = firstWorker (expected 0, the single block)
     *   [3] = firstStage (expected the stage passed to the barrier)
     *   [4] = firstWidth (expected 32)
     */
    struct DivergentKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            std::uint32_t barrierId,
            std::uint32_t stage,
            std::uint32_t activeCount,
            void* state,
            void* results) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            auto* diag = static_cast<graph::BarrierDiagnosticsState*>(state);
            auto* out = static_cast<std::uint32_t*>(results);
            graph::WarpGroup<Acc> warp{acc};
            graph::BarrierDiagnosticsConfig cfg;
            cfg.enabled = true;
            cfg.skipOnDivergence = true;
            auto diagnostics = graph::WarpBarrierDiagnostics<Acc>::make(diag, cfg);
            graph::WarpBarrier<Acc> barrier{diagnostics};
            constexpr std::uint32_t blockThreads = 32u;
            // Only a subset of the lanes calls the barrier. The diagnostic's
            // first-active-lane logic records one entry per divergent group;
            // the skip lets the kernel finish instead of deadlocking.
            if(warp.laneIndex() < activeCount)
                (void) barrier.barrier(acc, blockThreads, warp, barrierId, stage, 0u);
            if(warp.isLeader())
            {
                out[0] = static_cast<std::uint32_t>(diag->divergenceCount);
                out[1] = diag->firstBarrierId;
                out[2] = diag->firstWorker;
                out[3] = diag->firstStage;
                out[4] = diag->firstWidth;
            }
        }
    };

    void launchOneWarp(auto& device, auto&& kernel, auto&&... args)
    {
        auto const threadSpec = alpaka::onHost::ThreadSpec{
            alpaka::Vec<std::uint32_t, 1u>{1u},
            alpaka::Vec<std::uint32_t, 1u>{32u},
            alpaka::exec::gpuCuda};
        auto queue = device.makeQueue();
        queue.enqueue(threadSpec, kernel, args...);
        alpaka::onHost::wait(queue);
    }

    auto freshDiagnostics(auto& device, auto& queue)
    {
        auto state = alpaka::onHost::alloc<graph::BarrierDiagnosticsState>(device, 1u);
        auto hostState = alpaka::onHost::allocHost<graph::BarrierDiagnosticsState>(1u);
        auto* hs = alpaka::onHost::data(hostState);
        std::memset(hs, 0, sizeof(graph::BarrierDiagnosticsState));
        hs->firstBarrierId = ~std::uint32_t{0u};
        hs->firstWorker = ~std::uint32_t{0u};
        hs->firstStage = ~std::uint32_t{0u};
        alpaka::onHost::memcpy(queue, state, hostState);
        alpaka::onHost::wait(queue);
        return std::make_tuple(std::move(state), std::move(hostState));
    }
} // namespace

TEST_CASE("barrier diagnostics: a converged barrier records no divergence", "[graph][barrier][diag][cuda]")
{
    auto device = makeCudaDevice();
    auto queue = device.makeQueue();
    auto [state, hostState] = freshDiagnostics(device, queue);
    auto results = alpaka::onHost::alloc<std::uint32_t>(device, 2u);
    auto hostResults = alpaka::onHost::allocHost<std::uint32_t>(2u);

    launchOneWarp(
        device,
        ConvergedKernel{},
        std::uint32_t{42u},
        alpaka::onHost::data(state),
        alpaka::onHost::data(results));

    alpaka::onHost::memcpy(queue, hostResults, results);
    alpaka::onHost::memcpy(queue, hostState, state);
    alpaka::onHost::wait(queue);

    auto const* out = alpaka::onHost::data(hostResults);
    auto const* diag = alpaka::onHost::data(hostState);
    CHECK(out[0] == 0u);
    CHECK(out[1] == ~std::uint32_t{0u});
    CHECK(diag->divergenceCount == 0u);
    CHECK(diag->firstBarrierId == ~std::uint32_t{0u});
    std::printf(
        "[barrier-diag] converged: divergences=%llu firstBarrierId=0x%x\n",
        static_cast<unsigned long long>(diag->divergenceCount),
        static_cast<unsigned>(diag->firstBarrierId));
    std::fflush(stdout);
}

TEST_CASE("barrier diagnostics: a divergent arrival is recorded and skipped", "[graph][barrier][diag][cuda]")
{
    auto device = makeCudaDevice();
    auto queue = device.makeQueue();

    // Active lanes 0..23 (24 of 32) call the barrier; the remaining 8 are
    // absent. The diagnostic must record one divergence with the expected
    // call site, and the skip must let the kernel finish.
    constexpr std::uint32_t kActive = 24u;
    constexpr std::uint32_t kBarrierId = 7u;
    constexpr std::uint32_t kStage = 9u;

    auto [state, hostState] = freshDiagnostics(device, queue);
    auto results = alpaka::onHost::alloc<std::uint32_t>(device, 5u);
    auto hostResults = alpaka::onHost::allocHost<std::uint32_t>(5u);

    launchOneWarp(
        device,
        DivergentKernel{},
        kBarrierId,
        kStage,
        kActive,
        alpaka::onHost::data(state),
        alpaka::onHost::data(results));

    alpaka::onHost::memcpy(queue, hostResults, results);
    alpaka::onHost::memcpy(queue, hostState, state);
    alpaka::onHost::wait(queue);

    auto const* out = alpaka::onHost::data(hostResults);
    auto const* diag = alpaka::onHost::data(hostState);
    std::printf(
        "[barrier-diag] divergent: count=%llu firstBarrierId=0x%x firstWorker=0x%x firstStage=0x%x "
        "firstMask=0x%x firstWidth=%u perWorker[0]=%u\n",
        static_cast<unsigned long long>(diag->divergenceCount),
        static_cast<unsigned>(diag->firstBarrierId),
        static_cast<unsigned>(diag->firstWorker),
        static_cast<unsigned>(diag->firstStage),
        static_cast<unsigned>(diag->firstMask),
        static_cast<unsigned>(diag->firstWidth),
        static_cast<unsigned>(diag->perWorkerDivergences[0]));
    std::fflush(stdout);

    // The divergence was recorded.
    CHECK(diag->divergenceCount >= 1u);
    CHECK(diag->firstBarrierId == kBarrierId);
    CHECK(diag->firstWorker == 0u);
    CHECK(diag->firstStage == kStage);
    CHECK(diag->firstWidth == 32u);
    // The first record's mask is the active subset (lanes 0..kActive-1).
    CHECK(diag->firstMask == ((std::uint32_t{1u} << kActive) - 1u));
    CHECK(diag->perWorkerDivergences[0] >= 1u);
    // The kernel finished (the skip let it through); the result words are the
    // ones the leader wrote.
    CHECK(out[0] == diag->divergenceCount);
    CHECK(out[1] == kBarrierId);
    CHECK(out[2] == 0u);
    CHECK(out[3] == kStage);
    CHECK(out[4] == 32u);
}

#endif // GRAPH_HAS_CUDA
