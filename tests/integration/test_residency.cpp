// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/config.hpp>
#include <graph/detail/launch.hpp>
#include <graph/detail/rendezvous.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace
{
    struct Snapshot
    {
        std::uint64_t arrived{0};
        std::uint64_t expected{0};
        std::uint64_t launchCount{0};
        std::uint64_t allArrived{0};
        std::uint64_t completed{0};
        std::uint64_t firstClock{0};
        std::uint64_t lastClock{0};
        std::uint64_t minObserved{0};
    };

    Snapshot snapshotOf(std::uint64_t const* host, std::uint32_t capacity)
    {
        Snapshot result;
        result.arrived = host[graph::RendezvousLayout::arrived];
        result.expected = host[graph::RendezvousLayout::expected];
        result.launchCount = host[graph::RendezvousLayout::launchCount];
        result.allArrived = host[graph::RendezvousLayout::allArrived];
        result.completed = host[graph::RendezvousLayout::completed];
        result.firstClock = host[graph::RendezvousLayout::firstClock];
        result.lastClock = host[graph::RendezvousLayout::lastClock];

        std::uint64_t minObserved = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t const recorded = std::min<std::uint64_t>(result.arrived, capacity);
        for(std::uint64_t index = 0u; index < recorded; ++index)
            minObserved = std::min(
                minObserved,
                host[graph::RendezvousLayout::observedOffset(capacity) + static_cast<std::uint32_t>(index)]);
        result.minObserved = recorded == 0u ? 0u : minObserved;
        return result;
    }

    double milliseconds(std::chrono::steady_clock::duration duration)
    {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    void printGeometry(
        char const* backend,
        graph::LaunchEnvironment const& environment,
        graph::LaunchGeometry const& g)
    {
        std::printf(
            "[residency] backend=%s geometry blocks=%u threadsPerBlock=%u warpsPerBlock=%u workerCount=%u "
            "smCount=%u residentsPerSm=%u registersPerThread=%d captured=%d\n",
            backend,
            static_cast<unsigned>(g.blocks),
            static_cast<unsigned>(g.threadsPerBlock),
            static_cast<unsigned>(g.warpsPerBlock),
            static_cast<unsigned>(g.workerCount),
            static_cast<unsigned>(g.smCount),
            static_cast<unsigned>(g.residentsPerSm),
            environment.props.registersPerThreadHint,
            static_cast<int>(graph::kernelRegistersCaptured));
        std::fflush(stdout);
    }

    void printSnapshot(char const* backend, Snapshot const& snapshot, bool singleWave, double wallMs)
    {
        std::printf(
            "[residency] backend=%s arrived=%llu/%llu allArrived=%llu completed=%llu minObserved=%llu "
            "launchCount=%llu hostWallMs=%.3f firstClock=%llu lastClock=%llu singleWave=%d\n",
            backend,
            static_cast<unsigned long long>(snapshot.arrived),
            static_cast<unsigned long long>(snapshot.expected),
            static_cast<unsigned long long>(snapshot.allArrived),
            static_cast<unsigned long long>(snapshot.completed),
            static_cast<unsigned long long>(snapshot.minObserved),
            static_cast<unsigned long long>(snapshot.launchCount),
            wallMs,
            static_cast<unsigned long long>(snapshot.firstClock),
            static_cast<unsigned long long>(snapshot.lastClock),
            static_cast<int>(singleWave));
        std::fflush(stdout);
        std::printf(
            "[residency] backend=%s note=firstClock/lastClock are per-SM clock64 samples and are not comparable "
            "across blocks; hostWallMs is the arrival-time measurement\n",
            backend);
        std::fflush(stdout);
    }

    /** Allocate the state buffer, seed the expected count, and copy it over. */
    template<typename T_Device, typename T_Queue>
    auto prepareState(T_Device& device, T_Queue& queue, std::uint32_t capacity, std::uint32_t expected)
    {
        auto deviceState = alpaka::onHost::alloc<std::uint64_t>(device, graph::RendezvousLayout::wordCount(capacity));
        auto hostState = alpaka::onHost::allocHost<std::uint64_t>(graph::RendezvousLayout::wordCount(capacity));
        for(std::uint32_t index = 0u; index < graph::RendezvousLayout::wordCount(capacity); ++index)
            hostState[index] = 0u;
        hostState[graph::RendezvousLayout::expected] = expected;
        alpaka::onHost::memcpy(queue, deviceState, hostState);
        alpaka::onHost::wait(queue);
        return deviceState;
    }

    /** Host backend: progress and the CPU policy, explicitly not residency evidence. */
    void runHostRendezvous()
    {
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        REQUIRE(selector.getDeviceCount() > 0u);
        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();

        auto const environment = graph::queryLaunchEnvironment(device);
        auto const geometry = graph::computeLaunchGeometry(environment.props, graph::LaunchPolicy{});
        auto const validation = graph::validateSingleWave(geometry, environment.props, environment.limits);
        printGeometry("cpu", environment, geometry);

        REQUIRE(geometry.workerCount > 0u);
        REQUIRE(geometry.warpsPerBlock == 1u);
        REQUIRE(geometry.threadsPerBlock == 1u);

        auto deviceState = prepareState(device, queue, geometry.workerCount, geometry.workerCount);
        auto hostState
            = alpaka::onHost::allocHost<std::uint64_t>(graph::RendezvousLayout::wordCount(geometry.workerCount));

        auto const executor = alpaka::onHost::defaultExecutor(device);
        // CPU deadline is an iteration budget; it bounds a serialized executor.
        constexpr std::uint64_t cpuDeadlineIterations = 2'000'000u;
        auto const started = std::chrono::steady_clock::now();
        queue.enqueue(
            alpaka::onHost::ThreadSpec{geometry.blocks, geometry.threadsPerBlock, executor},
            graph::RendezvousKernel{},
            alpaka::onHost::data(deviceState),
            geometry.workerCount,
            cpuDeadlineIterations);
        alpaka::onHost::wait(queue);
        auto const finished = std::chrono::steady_clock::now();

        alpaka::onHost::memcpy(queue, hostState, deviceState);
        alpaka::onHost::wait(queue);

        Snapshot const snapshot = snapshotOf(alpaka::onHost::data(hostState), geometry.workerCount);
        printSnapshot("cpu", snapshot, validation.singleWave, milliseconds(finished - started));

        CAPTURE(
            snapshot.arrived,
            snapshot.expected,
            snapshot.completed,
            snapshot.launchCount,
            geometry.workerCount,
            environment.props.multiprocessors);

        // CPU policy: one single-thread worker per visible core.
        CHECK(geometry.workerCount == static_cast<std::uint32_t>(environment.props.multiprocessors));
        // Progress: at least one worker entered the pool and the kernel ran once.
        CHECK(snapshot.arrived > 0u);
        CHECK(snapshot.arrived <= geometry.workerCount);
        CHECK(snapshot.launchCount == 1u);
        // Not a co-residency claim: the CPU executor may run the blocks in
        // several batches, so `completed` is reported but not asserted.
        std::printf("[residency] backend=cpu note=progress-and-policy-only; this is not GPU residency evidence\n");
        std::fflush(stdout);
    }

#if GRAPH_HAS_CUDA
    void runCudaRendezvous()
    {
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            SKIP("CUDA backend was built but no NVIDIA GPU is present; residency rendezvous skipped");

        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();

        auto const environment = graph::queryLaunchEnvironment(device);
        auto const geometry = graph::computeLaunchGeometry(environment.props, graph::LaunchPolicy{});
        auto const validation = graph::validateSingleWave(geometry, environment.props, environment.limits);
        printGeometry("cuda", environment, geometry);

        CAPTURE(
            geometry.blocks,
            geometry.threadsPerBlock,
            geometry.workerCount,
            geometry.residentsPerSm,
            environment.props.registersPerThreadHint,
            graph::kernelRegistersCaptured,
            validation.reason);
        CHECK(validation.singleWave);

        auto deviceState = prepareState(device, queue, geometry.workerCount, geometry.workerCount);
        auto hostState
            = alpaka::onHost::allocHost<std::uint64_t>(graph::RendezvousLayout::wordCount(geometry.workerCount));

        auto const executor = alpaka::onHost::defaultExecutor(device);
        // At the measured 1.44 GHz clock this is about 50 ms per worker. A later
        // wave can only start after the earlier blocks leave at this deadline.
        constexpr std::uint64_t cudaDeadlineCycles = 72'000'000u;
        auto const started = std::chrono::steady_clock::now();
        queue.enqueue(
            alpaka::onHost::ThreadSpec{geometry.blocks, geometry.threadsPerBlock, executor},
            graph::RendezvousKernel{},
            alpaka::onHost::data(deviceState),
            geometry.workerCount,
            cudaDeadlineCycles);
        alpaka::onHost::wait(queue);
        auto const finished = std::chrono::steady_clock::now();

        alpaka::onHost::memcpy(queue, hostState, deviceState);
        alpaka::onHost::wait(queue);

        Snapshot const snapshot = snapshotOf(alpaka::onHost::data(hostState), geometry.workerCount);
        double const wallMs = milliseconds(finished - started);
        printSnapshot("cuda", snapshot, validation.singleWave, wallMs);

        CAPTURE(
            snapshot.arrived,
            snapshot.expected,
            snapshot.allArrived,
            snapshot.completed,
            snapshot.minObserved,
            snapshot.launchCount,
            wallMs);

        CHECK(snapshot.expected == geometry.workerCount);
        // All configured workers entered the pool.
        CHECK(snapshot.arrived == geometry.workerCount);
        CHECK(snapshot.allArrived == 1u);
        // Only a real simultaneous rendezvous increments `completed` for every
        // worker; sequential waves leave earlier workers below the expected count.
        CHECK(snapshot.completed == geometry.workerCount);
        CHECK(snapshot.minObserved >= geometry.workerCount);
        // Exactly one kernel launch.
        CHECK(snapshot.launchCount == 1u);
        // The observed worker count is the geometry's worker count.
        CHECK(snapshot.arrived == graph::computeLaunchGeometry(environment.props, graph::LaunchPolicy{}).workerCount);
        // Sanity bound: an all-wave rendezvous on the A30 completes in tens of
        // microseconds; this only catches a pathological multi-wave launch.
        CHECK(wallMs < 1'000.0);
    }
#endif
} // namespace

TEST_CASE("residency persistent workers rendezvous on the host backend", "[graph][residency][cpu]")
{
    runHostRendezvous();
}

#if GRAPH_HAS_CUDA
TEST_CASE(
    "residency persistent workers rendezvous on CUDA within the deadlined single wave",
    "[graph][residency][cuda]")
{
    runCudaRendezvous();
}
#endif
