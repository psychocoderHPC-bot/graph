// SPDX-FileCopyrightText: 2025 René Widera
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/config.hpp>

// Pull in the public primitives when this detail header is included on its own.
// system_atomic.hpp includes this file at its end; the include guard makes the
// cyclic include a no-op in that direction.
#include <graph/system_atomic.hpp>

#if GRAPH_HAS_CUDA
#    include <cuda_runtime.h>

#    include <cuda/atomic>
#endif

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

/** @file
 *
 * Capability probe and ingress-transport selection for the system-atomic
 * adapter.
 *
 * CUDA exposes allocation/atomics capabilities that do not line up with the
 * measured behaviour: on the HAL A30 `cudaDevAttrHostNativeAtomicSupported`
 * reads 0 while `atomicAdd_system` on a mapped pinned allocation was a true
 * atomic. The probe therefore records the attributes, and
 * `verifyMappedSystemAtomics()` performs a live CPU/GPU RMW round-trip on a
 * mapped allocation. Callers must not enable the raw RMW transport from the
 * attribute alone.
 */
namespace graph
{
    /** System memory capabilities of one device.
     *
     * `deviceSystemScopeRmwNative` is the attribute-derived (conservative)
     * capability in `inquireSystemMemoryCapabilities()`. The
     * `inquireVerifiedSystemMemoryCapabilities()` convenience function replaces
     * it with the measured effective value after a successful live round-trip.
     */
    struct SystemMemoryCapabilities
    {
        bool deviceSystemScopeRmwNative = false;
        bool hostNativeAtomicSupported = false;
        bool concurrentManagedAccess = false;
        bool mappedPinnedAllocation = false;
        bool mappedSystemAtomicVerified = false;
    };

    /** Result of a live accelerator/host read-modify-write round-trip on mapped memory. */
    struct SystemAtomicVerification
    {
        bool mappedPinnedAllocation = false;
        bool orderedAccessVerified = false;
        bool nativeSystemRmwVerified = false;
        std::uint64_t expected = 0;
        std::uint64_t observedOrdered = 0;
        std::uint64_t observedNative = 0;
        std::string detail;
    };

    /** Ingress transport chosen for a host/GPU SPSC ring. */
    enum class IngressTransport
    {
        /// No correct transport is available; startup must report the gap.
        Unsupported,
        /// System-scope acquire loads and release stores (preferred default).
        OrderedSystemAtomic,
        /// Raw `::atomicAdd_system` doorbell, opt-in only after live verification.
        NativeSystemRmw
    };

    inline constexpr std::string_view toString(IngressTransport transport) noexcept
    {
        switch(transport)
        {
        case IngressTransport::Unsupported:
            return "Unsupported";
        case IngressTransport::OrderedSystemAtomic:
            return "OrderedSystemAtomic";
        case IngressTransport::NativeSystemRmw:
            return "NativeSystemRmw";
        }
        return "Unknown";
    }
} // namespace graph

namespace graph::detail
{
    /** Kernel used by the live capability probe.
     *
     * The ordered counter uses `graph::atomicAdd(..., scope::System)`; the raw
     * counter uses `graph::atomicAddSystemNative(..., scope::System)`. The host
     * performs the same number of adds, so the exact expected sum checks that
     * both CPU and GPU atomics target the same physical words.
     */
    struct VerifyMappedRmwKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto counter,
            std::uint64_t iterations) const
        {
            auto* ptr = &counter[0];
            for(std::uint64_t i = 0u; i < iterations; ++i)
            {
                graph::atomicAdd(acc, ptr, std::uint64_t{1}, alpaka::onAcc::scope::system);
                graph::atomicAddSystemNative(acc, ptr + 1, std::uint64_t{1}, alpaka::onAcc::scope::system);
            }
        }
    };
} // namespace graph::detail

namespace graph::detail::cpu
{
    template<typename T_Device>
    SystemMemoryCapabilities inquireCapabilities([[maybe_unused]] T_Device const& device)
    {
        SystemMemoryCapabilities caps;
        bool const lockFree = std::atomic_ref<std::uint64_t>::is_always_lock_free;
        caps.deviceSystemScopeRmwNative = lockFree;
        // A CPU device is the host: native host atomics and mapped host memory
        // are the ordinary host memory model.
        caps.hostNativeAtomicSupported = lockFree;
        caps.concurrentManagedAccess = false;
        caps.mappedPinnedAllocation = true;
        caps.mappedSystemAtomicVerified = false;
        return caps;
    }

    template<typename T_Device>
    SystemAtomicVerification verifyMappedSystemAtomics(T_Device& device)
    {
        SystemAtomicVerification result;
        result.detail = "CPU host memory";

        auto counter = alpaka::onHost::allocMapped<std::uint64_t>(device, alpaka::Vec<std::uint32_t, 1u>{2u});
        auto* ptr = alpaka::onHost::data(counter);
        ptr[0] = std::uint64_t{0};
        ptr[1] = std::uint64_t{0};

        constexpr std::uint64_t iterations = 1000u;
        auto queue = device.makeQueue();
        queue.enqueue(
            alpaka::onHost::FrameSpec{1u, 1u, alpaka::exec::cpuSerial},
            graph::detail::VerifyMappedRmwKernel{},
            counter,
            iterations);
        alpaka::onHost::wait(queue);

        // One host RMW after the device work completes the init/device/host
        // round-trip on the same physical words.
        std::atomic_ref<std::uint64_t> orderedRef(ptr[0]);
        std::atomic_ref<std::uint64_t> nativeRef(ptr[1]);
        orderedRef.fetch_add(std::uint64_t{1}, std::memory_order_relaxed);
        nativeRef.fetch_add(std::uint64_t{1}, std::memory_order_relaxed);

        result.expected = iterations + 1u;
        result.observedOrdered = ptr[0];
        result.observedNative = ptr[1];
        result.mappedPinnedAllocation = true;
        result.orderedAccessVerified = result.observedOrdered == result.expected;
        result.nativeSystemRmwVerified = result.observedNative == result.expected;
        return result;
    }
} // namespace graph::detail::cpu

namespace graph::detail::cuda
{
    template<typename T_Device>
    SystemMemoryCapabilities inquireCapabilities(T_Device const& device)
    {
#if GRAPH_HAS_CUDA
        int const deviceIndex = alpaka::onHost::getNativeHandle(device);
        auto query = [deviceIndex](cudaDeviceAttr attribute) -> int
        {
            int value = 0;
            cudaError_t const status = cudaDeviceGetAttribute(&value, attribute, deviceIndex);
            if(status != cudaSuccess)
                throw std::runtime_error(
                    std::string{"graph: cudaDeviceGetAttribute failed: "} + cudaGetErrorString(status));
            return value;
        };

        SystemMemoryCapabilities caps;
        caps.hostNativeAtomicSupported = query(cudaDevAttrHostNativeAtomicSupported) != 0;
        caps.concurrentManagedAccess = query(cudaDevAttrConcurrentManagedAccess) != 0;
        caps.mappedPinnedAllocation = query(cudaDevAttrCanMapHostMemory) != 0;
        // CUDA has no separate attribute for system-scope RMW on host memory.
        // The closest documented attribute is host-native atomics; keep the
        // attribute-derived value conservative and verify the live allocation.
        caps.deviceSystemScopeRmwNative = caps.hostNativeAtomicSupported;
        caps.mappedSystemAtomicVerified = false;
        return caps;
#else
        // The CUDA backend is not part of this build. Report every capability as
        // absent instead of silently selecting a transport.
        [[maybe_unused]] T_Device const& unused = device;
        return SystemMemoryCapabilities{};
#endif
    }

    template<typename T_Device>
    SystemAtomicVerification verifyMappedSystemAtomics(T_Device& device)
    {
#if GRAPH_HAS_CUDA
        SystemAtomicVerification result;
        result.detail = "CUDA mapped pinned memory";

        auto counter = alpaka::onHost::allocMapped<std::uint64_t>(device, alpaka::Vec<std::uint32_t, 1u>{2u});
        auto* ptr = alpaka::onHost::data(counter);
        ptr[0] = std::uint64_t{0};
        ptr[1] = std::uint64_t{0};

        constexpr std::uint64_t iterations = 1000u;
        constexpr std::uint64_t blocks = 2u;
        constexpr std::uint64_t threads = 32u;
        auto queue = device.makeQueue();
        queue.enqueue(
            alpaka::onHost::ThreadSpec{
                static_cast<std::uint32_t>(blocks),
                static_cast<std::uint32_t>(threads),
                alpaka::exec::gpuCuda},
            graph::detail::VerifyMappedRmwKernel{},
            counter,
            iterations);
        alpaka::onHost::wait(queue);

        // A host RMW after the device work completes the round-trip and proves
        // that both sides can reach the same physical words.
        ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> orderedRef(ptr[0]);
        ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> nativeRef(ptr[1]);
        orderedRef.fetch_add(std::uint64_t{1}, ::cuda::std::memory_order_relaxed);
        nativeRef.fetch_add(std::uint64_t{1}, ::cuda::std::memory_order_relaxed);

        result.expected = blocks * threads * iterations + 1u;
        result.observedOrdered = ptr[0];
        result.observedNative = ptr[1];
        result.mappedPinnedAllocation = true;
        result.orderedAccessVerified = result.observedOrdered == result.expected;
        result.nativeSystemRmwVerified = result.observedNative == result.expected;
        return result;
#else
        [[maybe_unused]] T_Device const& unused = device;
        throw std::runtime_error("graph: the CUDA backend is not enabled; mapped system atomics cannot be verified");
#endif
    }
} // namespace graph::detail::cuda

namespace graph
{
    /** Query the device's system memory attributes.
     *
     * This only reads attributes. It never enables the raw RMW transport and
     * always returns `mappedSystemAtomicVerified == false`.
     *
     * Does NOT guarantee: that a mapped allocation supports system RMW. Use
     * `verifyMappedSystemAtomics()` on the live allocation.
     */
    template<typename T_Device>
    SystemMemoryCapabilities inquireSystemMemoryCapabilities(T_Device const& device)
    {
        using Api = ALPAKA_TYPEOF(device.getApi());
        if constexpr(detail::isCudaApi<Api>)
            return detail::cuda::inquireCapabilities(device);
        else if constexpr(detail::isHostApi<Api>)
            return detail::cpu::inquireCapabilities(device);
        else
            static_assert(
                detail::alwaysFalse<T_Device>,
                "graph: system memory capabilities are implemented for the CPU and NVIDIA CUDA backends only");
    }

    /** Perform a live accelerator/host RMW round-trip on a mapped allocation.
     *
     * The function allocates a mapped pinned counter pair, runs a multi-threaded
     * accelerator RMW burst through the ordered system RMW and through the raw
     * `::atomicAdd_system` path, then one host RMW after the device work
     * completes, and reports the measured values. A result of
     * `nativeSystemRmwVerified == true` is measured behaviour on this device,
     * not a portability contract.
     *
     * This verifies the RMW path on the allocation; it does not claim that
     * concurrent host and device RMWs on mapped memory are atomic. That needs
     * `hostNativeAtomicSupported` (mapped) or managed memory with
     * `concurrentManagedAccess`; the capability struct reports both.
     */
    template<typename T_Device>
    SystemAtomicVerification verifyMappedSystemAtomics(T_Device& device)
    {
        using Api = ALPAKA_TYPEOF(device.getApi());
        if constexpr(detail::isCudaApi<Api>)
            return detail::cuda::verifyMappedSystemAtomics(device);
        else if constexpr(detail::isHostApi<Api>)
            return detail::cpu::verifyMappedSystemAtomics(device);
        else
            static_assert(
                detail::alwaysFalse<T_Device>,
                "graph: mapped system atomics are implemented for the CPU and NVIDIA CUDA backends only");
    }

    /** Attribute probe followed by a live verification.
     *
     * The returned `deviceSystemScopeRmwNative` is the effective value:
     * attribute OR live measurement. `mappedSystemAtomicVerified` records
     * whether the raw RMW round-trip succeeded.
     */
    template<typename T_Device>
    SystemMemoryCapabilities inquireVerifiedSystemMemoryCapabilities(T_Device& device)
    {
        SystemMemoryCapabilities caps = inquireSystemMemoryCapabilities(device);
        SystemAtomicVerification const verification = verifyMappedSystemAtomics(device);
        caps.mappedPinnedAllocation = caps.mappedPinnedAllocation && verification.mappedPinnedAllocation;
        caps.mappedSystemAtomicVerified = verification.nativeSystemRmwVerified;
        caps.deviceSystemScopeRmwNative = caps.deviceSystemScopeRmwNative || verification.nativeSystemRmwVerified;
        return caps;
    }

    /** Select the ingress transport for a host/GPU SPSC ring.
     *
     * Policy:
     * 1. Without a mapped pinned allocation no correct cross-system transport
     *    exists; return `Unsupported` so startup reports the missing capability.
     * 2. Prefer the ordered acquire/release transport. It is the contract-backed
     *    path and the measured fast path (1.6 M msg/s ingress on the HAL A30).
     * 3. The raw RMW doorbell is opt-in through @p allowNativeSystemRmw and is
     *    only selected when the live round-trip verified it. A requested but
     *    unverified raw path returns `Unsupported`; it never silently degrades.
     *
     * The raw path is measured behaviour, not a contract, and costs about
     * 597 ns per operation, so it is unsuitable for per-job publication.
     */
    inline IngressTransport selectIngressTransport(
        SystemMemoryCapabilities const& caps,
        bool allowNativeSystemRmw = false)
    {
        if(!caps.mappedPinnedAllocation)
            return IngressTransport::Unsupported;

        if(allowNativeSystemRmw)
        {
            if(caps.deviceSystemScopeRmwNative && caps.mappedSystemAtomicVerified)
                return IngressTransport::NativeSystemRmw;
            return IngressTransport::Unsupported;
        }

        return IngressTransport::OrderedSystemAtomic;
    }
} // namespace graph
