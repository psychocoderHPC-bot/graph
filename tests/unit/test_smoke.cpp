// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/config.hpp>

#include <cstdint>

static_assert(GRAPH_VERSION_MAJOR == 0);
static_assert(GRAPH_VERSION_MINOR == 1);
static_assert(GRAPH_VERSION_PATCH == 0);

namespace
{
    struct FillKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::Vec<std::uint32_t, 1u> extent,
            std::uint32_t value) const
        {
            for(auto [index] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{extent}))
            {
                out[index] = value;
            }
        }
    };
} // namespace

TEST_CASE("graph smoke test runs a kernel on the configured backend", "[graph][smoke]")
{
#if GRAPH_HAS_CUDA
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto const executor = alpaka::exec::gpuCuda;
#else
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
    auto const executor = alpaka::exec::cpuSerial;
#endif

    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    REQUIRE(selector.getDeviceCount() > 0u);

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    constexpr std::uint32_t size = 128u;
    constexpr std::uint32_t value = 42u;

    auto hostBuffer = alpaka::onHost::allocHost<std::uint32_t>(size);
    for(std::uint32_t i = 0u; i < size; ++i)
    {
        hostBuffer[i] = 0u;
    }

    auto deviceBuffer = alpaka::onHost::allocLike(device, hostBuffer);
    auto const frameSpec = alpaka::onHost::FrameSpec{32u, 32u, executor};
    queue.enqueue(frameSpec, FillKernel{}, deviceBuffer, alpaka::Vec<std::uint32_t, 1u>{size}, value);
    alpaka::onHost::memcpy(queue, hostBuffer, deviceBuffer);
    alpaka::onHost::wait(queue);

    for(std::uint32_t i = 0u; i < size; ++i)
    {
        REQUIRE(hostBuffer[i] == value);
    }
}
