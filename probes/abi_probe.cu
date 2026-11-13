// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

/** Device-side ABI probe.
 *
 * Takes the address of an instantiated `OperationFn<T_Acc>` and stores it in a
 * `__device__` array, then reloads and calls it from device code. This proves
 * that the functor ABI is a usable device function-pointer type and that the
 * project builds with relocatable device code. This translation unit is only
 * built for the CUDA configuration.
 */

#include <alpaka/alpaka.hpp>

#include <graph/detail/operation.hpp>

#include <cstdint>
#include <type_traits>

namespace
{
    template<typename T_Acc>
    struct DeviceAbiProbe
    {
        ALPAKA_FN_ACC static void operation(
            T_Acc const&,
            graph::JobDescriptor const&,
            graph::PacketAccessor const&) noexcept
        {
        }
    };

    // Device-resident table of the instantiated ABI pointer type.
    template<typename T_Acc>
    __device__ graph::OperationFn<T_Acc> deviceProbeFunctions[1] = {&DeviceAbiProbe<T_Acc>::operation};

    struct ProbeKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Fn = graph::OperationFn<Acc>;

            static_assert(std::is_same_v<Fn, std::remove_reference_t<decltype(deviceProbeFunctions<Acc>[0])>>);
            static_assert(std::is_pointer_v<Fn>);

            // Reload through the device array and store a device-side address
            // back into it, so the pointer is formed in device code.
            auto const function = deviceProbeFunctions<Acc>[0];
            deviceProbeFunctions<Acc>[0] = &DeviceAbiProbe<Acc>::operation;

            if(function != nullptr)
                function(acc, graph::JobDescriptor{}, graph::PacketAccessor{});
        }
    };
} // namespace

int main()
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
        return 0;

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto const threadSpec = alpaka::onHost::ThreadSpec{1u, 32u, alpaka::exec::gpuCuda};
    queue.enqueue(threadSpec, ProbeKernel{});
    alpaka::onHost::wait(queue);
    return 0;
}
