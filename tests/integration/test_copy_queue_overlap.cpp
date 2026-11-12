// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/config.hpp>
#include <graph/detail/copy_queue.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace
{
    auto milliseconds(std::chrono::steady_clock::duration duration) -> double
    {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    /** Host-backend round trip: the copy queue exists and copies both ways. */
    void runHostRoundTrip()
    {
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        REQUIRE(selector.getDeviceCount() > 0u);

        auto device = selector.makeDevice(0);
        graph::CopyQueue<decltype(device)> copyQueue{device};

        // The object owns a usable queue handle on the device.
        CHECK(copyQueue.queue().get() != nullptr);
        CHECK(copyQueue.device().get() != nullptr);

        constexpr std::size_t count = 16u;
        auto hostInput = alpaka::onHost::allocHost<std::uint32_t>(count);
        for(std::size_t i = 0u; i < count; ++i)
            hostInput[i] = static_cast<std::uint32_t>(1000u + i);

        auto deviceBuffer = alpaka::onHost::alloc<std::uint32_t>(device, count);
        auto hostOutput = alpaka::onHost::allocHost<std::uint32_t>(count);
        for(std::size_t i = 0u; i < count; ++i)
            hostOutput[i] = 0u;

        auto toDevice = copyQueue.asyncCopyToDevice(deviceBuffer, hostInput);
        toDevice.wait();
        CHECK(toDevice.ready());

        auto toHost = copyQueue.asyncCopyToHost(hostOutput, deviceBuffer);
        toHost.wait();
        CHECK(toHost.ready());

        for(std::size_t i = 0u; i < count; ++i)
            CHECK(hostOutput[i] == hostInput[i]);
    }

#if GRAPH_HAS_CUDA
    /** A few-millisecond busy loop so the worker queue stays occupied. */
    struct BusyKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, std::uint32_t iterations, std::uint32_t* sink) const
        {
            auto const lane = alpaka::onAcc::warp::getLaneIdx(acc);
            std::uint32_t value = lane + 12345u;
            for(std::uint32_t i = 0u; i < iterations; ++i)
                value = value * 1'664'525u + 1'013'904'223u;
            sink[lane] = value;
        }
    };

    auto makeCudaDevice()
    {
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
            SKIP("CUDA backend was built but no NVIDIA GPU is present; overlap test skipped");
        return selector.makeDevice(0);
    }

    /** Copy on the copy queue must finish while the worker queue is still busy. */
    void runCudaOverlap()
    {
        auto device = makeCudaDevice();
        auto const warpSize = device.getDeviceProperties().warpSize;

        auto workerQueue = device.makeQueue();
        graph::CopyQueue<decltype(device)> copyQueue{device};

        constexpr std::size_t count = 64u;
        auto hostInput = copyQueue.template allocateStaging<std::uint32_t>(count);
        for(std::size_t i = 0u; i < count; ++i)
            hostInput[i] = static_cast<std::uint32_t>(0xA5A5'0000u + i);

        auto dataBuffer = alpaka::onHost::alloc<std::uint32_t>(device, count);
        auto sinkBuffer = alpaka::onHost::alloc<std::uint32_t>(device, warpSize);

        // Setup copy completes before the persistent worker starts.
        copyQueue.asyncCopyToDevice(dataBuffer, hostInput).wait();

        auto hostResult = copyQueue.template allocateStaging<std::uint32_t>(count);
        for(std::size_t i = 0u; i < count; ++i)
            hostResult[i] = 0u;

        auto workerEvent = workerQueue.makeEvent();
        auto const workerStart = std::chrono::steady_clock::now();
        // 30M dependent iterations per lane is tens of milliseconds on the A30.
        workerQueue.enqueue(
            alpaka::onHost::ThreadSpec{1u, warpSize, alpaka::exec::gpuCuda},
            BusyKernel{},
            30'000'000u,
            sinkBuffer.data());
        workerQueue.enqueue(workerEvent);
        bool const workerPendingAfterEnqueue = !workerEvent.isComplete();

        auto const copyStart = std::chrono::steady_clock::now();
        auto copyToken = copyQueue.asyncCopyToHost(hostResult, dataBuffer);
        copyToken.wait();
        auto const copyEnd = std::chrono::steady_clock::now();

        bool const copyCompleted = copyToken.ready();
        bool const workerPendingWhenCopyDone = !workerEvent.isComplete();

        alpaka::onHost::wait(workerQueue);
        auto const workerEnd = std::chrono::steady_clock::now();

        double const copyMs = milliseconds(copyEnd - copyStart);
        double const workerRemainingMs = milliseconds(workerEnd - copyEnd);
        double const workerTotalMs = milliseconds(workerEnd - workerStart);

        std::printf(
            "[copy_queue] cuda copy_completed=%d worker_pending_when_copy_done=%d "
            "worker_pending_after_enqueue=%d copy_ms=%.3f worker_remaining_ms=%.3f worker_total_ms=%.3f\n",
            static_cast<int>(copyCompleted),
            static_cast<int>(workerPendingWhenCopyDone),
            static_cast<int>(workerPendingAfterEnqueue),
            copyMs,
            workerRemainingMs,
            workerTotalMs);
        std::fflush(stdout);

        CAPTURE(copyCompleted, workerPendingWhenCopyDone, copyMs, workerRemainingMs, workerTotalMs);

        CHECK(copyCompleted);
        CHECK(workerPendingWhenCopyDone);
        for(std::size_t i = 0u; i < count; ++i)
            CHECK(hostResult[i] == static_cast<std::uint32_t>(0xA5A5'0000u + i));
    }

    /** Pageable host buffers must fail loudly instead of silently losing overlap. */
    void runCudaPageableRejection()
    {
        auto device = makeCudaDevice();
        graph::CopyQueue<decltype(device)> copyQueue{device};

        auto pageable = alpaka::onHost::allocHost<std::uint32_t>(4u);
        auto deviceBuffer = alpaka::onHost::alloc<std::uint32_t>(device, 4u);

        CHECK_THROWS_AS(copyQueue.asyncCopyToHost(pageable, deviceBuffer), std::invalid_argument);
        CHECK_THROWS_AS(copyQueue.asyncCopyToDevice(deviceBuffer, pageable), std::invalid_argument);

        // The mapped staging path is accepted.
        auto mapped = copyQueue.template allocateStaging<std::uint32_t>(4u);
        CHECK_NOTHROW(copyQueue.asyncCopyToDevice(deviceBuffer, mapped).wait());
    }
#endif
} // namespace

TEST_CASE("copy_queue round-trips a small buffer on the host backend", "[graph][copy_queue]")
{
    runHostRoundTrip();
}

#if GRAPH_HAS_CUDA
TEST_CASE("copy_queue overlaps a busy worker queue on CUDA", "[graph][copy_queue][cuda]")
{
    runCudaOverlap();
}

TEST_CASE("copy_queue rejects pageable host buffers on CUDA", "[graph][copy_queue][cuda]")
{
    runCudaPageableRejection();
}
#endif
