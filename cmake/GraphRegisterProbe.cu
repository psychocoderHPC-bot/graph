// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

/** Configure-time register probe.
 *
 * This translation unit instantiates the two persistent entry points through
 * real alpaka CUDA launches: `graph::detail::PoolKernel` (the pool's own
 * persistent worker, which sizes the launch geometry) and
 * `graph::RendezvousKernel` (the co-residency gate). `cmake/GraphRegisters.cmake`
 * compiles it with `-Xptxas -v` and takes the maximum reported register count,
 * so the launch geometry uses the allocation of the compiled entry point. The
 * probe is compiled, never run.
 *
 * `PoolKernel` is instantiated with `graph::detail::NullBinder`, the default
 * binder of `graph::startPool()`. A user binder adds its own `bind` body to the
 * startup phase; the capture is a property of the library's entry point, and a
 * higher user value must be passed through `PoolConfig::registerHint`.
 */

#include <alpaka/alpaka.hpp>

#include <graph/detail/launch.hpp>
#include <graph/detail/rendezvous.hpp>
#include <graph/detail/scheduler.hpp>
#include <graph/pool.hpp>

#include <cstdint>

int main()
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();

    std::uint32_t const warpSize = device.getDeviceProperties().warpSize;

    // The pool's own persistent kernel: one warp, default binder, empty args.
    graph::detail::PoolKernelArgs poolArgs{};
    queue.enqueue(
        alpaka::onHost::ThreadSpec{std::uint32_t{1u}, warpSize, alpaka::exec::gpuCuda},
        graph::detail::PoolKernel<graph::detail::NullBinder>{},
        poolArgs);

    // The co-residency rendezvous kernel.
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
