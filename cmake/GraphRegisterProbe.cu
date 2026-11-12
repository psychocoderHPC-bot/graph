// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

/** Configure-time register probe.
 *
 * This translation unit instantiates the persistent rendezvous kernel through a
 * real alpaka CUDA launch. `cmake/GraphRegisters.cmake` compiles it with
 * `-Xptxas -v` and parses the register count, so the launch geometry uses the
 * allocation of the compiled entry point. The probe is compiled, never run.
 */

#include <alpaka/alpaka.hpp>

#include <graph/detail/launch.hpp>
#include <graph/detail/rendezvous.hpp>

#include <cstdint>

int main()
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    std::uint32_t const warpSize = device.getDeviceProperties().warpSize;
    auto state = alpaka::onHost::alloc<std::uint64_t>(device, std::uint32_t{16u});

    queue.enqueue(
        alpaka::onHost::ThreadSpec{std::uint32_t{1u}, warpSize, alpaka::exec::gpuCuda},
        graph::RendezvousKernel{},
        alpaka::onHost::data(state),
        std::uint32_t{8u},
        std::uint64_t{1'000'000u});
    alpaka::onHost::wait(queue);
    return 0;
}
