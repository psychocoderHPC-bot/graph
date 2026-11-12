// SPDX-FileCopyrightText: 2025 René Widera
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/atomic_cpu.hpp>
#include <graph/detail/atomic_cuda.hpp>
#include <graph/detail/config.hpp>

#include <concepts>
#include <type_traits>

/** @file
 *
 * Project-owned system-scope atomic and ordered-access adapter.
 *
 * The pinned alpaka revision deliberately rejects `alpaka::onAcc::scope::System`
 * in `onAcc::atomicOp` (`onAcc/atomic.hpp:28,51`). This header supplies the
 * missing overload in `namespace graph` and forwards all block/device-scope
 * operations to the upstream API. Backend selection uses the accelerator API
 * tag (`ALPAKA_TYPEOF(acc.getApi())`) and `if constexpr`, mirroring alpaka's own
 * implementation style.
 *
 * Contracts:
 * - `graph::atomicAdd(..., scope::System)` is a system-scope relaxed RMW. It
 *   does not publish or acquire surrounding data.
 * - `graph::systemStoreRelease` / `graph::systemLoadAcquire` are the ordered
 *   publication pair. The release store orders every preceding write; the
 *   acquire load orders every following read.
 * - A fence and ordinary non-atomic accesses do not implement the protocol.
 *   Use the atomic load/store pair, optionally followed by the matching fence.
 * - A raw RMW must never be used as a load. Poll with the ordered load, or use
 *   the explicit `atomicAddSystemNative` doorbell only on a verified allocation.
 *
 * The host side has one matching primitive set in `namespace graph::host`.
 */
namespace graph
{
    namespace detail
    {
        template<typename... T>
        inline constexpr bool alwaysFalse = false;

        template<typename T_Api>
        inline constexpr bool isCudaApi = std::is_same_v<std::remove_cvref_t<T_Api>, alpaka::api::Cuda>;

        template<typename T_Api>
        inline constexpr bool isHostApi = std::is_same_v<std::remove_cvref_t<T_Api>, alpaka::api::Host>;
    } // namespace detail

    /** @name Block/device-scope forwarding
     *  The project keeps the upstream overload set intact. These overloads only
     *  make the calls available through `graph::` and forward verbatim.
     */
    ///@{
    template<typename T, typename T_Acc>
    constexpr T atomicAdd(T_Acc const& acc, T* const ptr, T const value)
    {
        return alpaka::onAcc::atomicAdd(acc, ptr, value, alpaka::onAcc::scope::device);
    }

    template<typename T, typename T_Acc, typename T_Scope>
    requires std::same_as<std::remove_cvref_t<T_Scope>, alpaka::onAcc::scope::Device>
             || std::same_as<std::remove_cvref_t<T_Scope>, alpaka::onAcc::scope::Block>
    constexpr T atomicAdd(T_Acc const& acc, T* const ptr, T const value, T_Scope const scope)
    {
        return alpaka::onAcc::atomicAdd(acc, ptr, value, scope);
    }

    ///@}

    /** System-scope atomic add, returning the previous value.
     *
     * Ordering contract: relaxed system-scope read-modify-write. It does not
     * order preceding writes before the RMW and does not order following reads
     * after it. Pair it with the fence functions when the algorithm needs that
     * ordering.
     *
     * Does NOT guarantee: publication of surrounding data, and no CPU/GPU
     * visibility unless the allocation is system-accessible (mapped pinned on
     * CUDA). The raw RMW transport additionally requires a verified allocation;
     * see `graph::verifyMappedSystemAtomics`.
     */
    template<typename T, typename T_Acc>
    constexpr T atomicAdd(T_Acc const& acc, T* const ptr, T const value, alpaka::onAcc::scope::System)
    {
        using Api = ALPAKA_TYPEOF(acc.getApi());
        if constexpr(detail::isCudaApi<Api>)
            return detail::cuda::systemFetchAdd(ptr, value);
        else if constexpr(detail::isHostApi<Api>)
            return detail::cpu::systemFetchAdd(ptr, value);
        else
            static_assert(
                detail::alwaysFalse<T_Acc>,
                "graph: system-scope atomics are implemented for the CPU and NVIDIA CUDA backends only");
    }

    /** Explicit opt-in raw `::atomicAdd_system` doorbell.
     *
     * This exists only for the rare, capability-verified doorbell path. The raw
     * intrinsic is measured behaviour, not a portability contract: on an A30,
     * `atomicAdd_system` on mapped pinned memory was a true atomic (256000 of
     * 256000 unique results) while `cudaDevAttrHostNativeAtomicSupported`
     * reported 0. It was also measured at ~597 ns per operation, so never use it
     * per job.
     *
     * Ordering contract: relaxed system-scope read-modify-write, like
     * `graph::atomicAdd(..., scope::System)`.
     *
     * Does NOT guarantee: support on the live allocation. Call
     * `graph::verifyMappedSystemAtomics` first and require
     * `SystemAtomicVerification::nativeSystemRmwVerified`.
     */
    template<typename T, typename T_Acc>
    constexpr T atomicAddSystemNative(T_Acc const& acc, T* const ptr, T const value, alpaka::onAcc::scope::System)
    {
        using Api = ALPAKA_TYPEOF(acc.getApi());
        if constexpr(detail::isCudaApi<Api>)
            return detail::cuda::nativeSystemFetchAdd(ptr, value);
        else if constexpr(detail::isHostApi<Api>)
            return detail::cpu::systemFetchAdd(ptr, value);
        else
            static_assert(
                detail::alwaysFalse<T_Acc>,
                "graph: the raw system RMW is implemented for the CPU and NVIDIA CUDA backends only");
    }

    /** System-scope acquire load.
     *
     * Ordering contract: every read or write after this load observes all
     * writes that the matching release store ordered before itself.
     *
     * Does NOT guarantee: atomicity for a value the producer wrote with a
     * non-atomic store, and no ordering without the matching release operation.
     */
    template<typename T, typename T_Acc>
    constexpr T systemLoadAcquire(T_Acc const& acc, T const* const ptr)
    {
        using Api = ALPAKA_TYPEOF(acc.getApi());
        if constexpr(detail::isCudaApi<Api>)
            return detail::cuda::systemLoadAcquire(ptr);
        else if constexpr(detail::isHostApi<Api>)
            return detail::cpu::systemLoadAcquire(ptr);
        else
            static_assert(
                detail::alwaysFalse<T_Acc>,
                "graph: system-scope atomics are implemented for the CPU and NVIDIA CUDA backends only");
    }

    /** System-scope release store.
     *
     * Ordering contract: every write before this store is visible to a thread
     * that later performs the matching acquire load.
     *
     * Does NOT guarantee: ordering of writes after the store, and no ordering
     * without the matching acquire operation.
     */
    template<typename T, typename T_Acc>
    constexpr void systemStoreRelease(T_Acc const& acc, T* const ptr, T const value)
    {
        using Api = ALPAKA_TYPEOF(acc.getApi());
        if constexpr(detail::isCudaApi<Api>)
            detail::cuda::systemStoreRelease(ptr, value);
        else if constexpr(detail::isHostApi<Api>)
            detail::cpu::systemStoreRelease(ptr, value);
        else
            static_assert(
                detail::alwaysFalse<T_Acc>,
                "graph: system-scope atomics are implemented for the CPU and NVIDIA CUDA backends only");
    }

    /** System-scope acquire fence on the accelerator.
     *
     * Ordering contract: all reads after the fence observe writes that were
     * released before the fence's paired observation. Pair it with a preceding
     * observation of the producer's release operation.
     */
    template<typename T_Acc>
    constexpr void systemFenceAcquire(T_Acc const& acc)
    {
        using Api = ALPAKA_TYPEOF(acc.getApi());
        if constexpr(detail::isCudaApi<Api>)
            detail::cuda::systemFenceAcquire();
        else if constexpr(detail::isHostApi<Api>)
            detail::cpu::systemFenceAcquire();
        else
            static_assert(
                detail::alwaysFalse<T_Acc>,
                "graph: system-scope fences are implemented for the CPU and NVIDIA CUDA backends only");
    }

    /** System-scope release fence on the accelerator.
     *
     * Ordering contract: all writes before the fence are ordered before a
     * following publication. Pair it with the publication operation.
     */
    template<typename T_Acc>
    constexpr void systemFenceRelease(T_Acc const& acc)
    {
        using Api = ALPAKA_TYPEOF(acc.getApi());
        if constexpr(detail::isCudaApi<Api>)
            detail::cuda::systemFenceRelease();
        else if constexpr(detail::isHostApi<Api>)
            detail::cpu::systemFenceRelease();
        else
            static_assert(
                detail::alwaysFalse<T_Acc>,
                "graph: system-scope fences are implemented for the CPU and NVIDIA CUDA backends only");
    }
} // namespace graph

// The capability probe and transport selection build on the primitives above.
// Includes at the end so this header remains the single public entry point.
#include <graph/detail/capabilities.hpp>
