// SPDX-FileCopyrightText: 2025 René Widera
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/core/common.hpp>

#include <graph/detail/config.hpp>

#include <concepts>
#include <type_traits>

#if GRAPH_HAS_CUDA
#    include <cuda_runtime.h>

#    include <cuda/atomic>
#    include <cuda/std/atomic>
#endif

/** @file
 *
 * NVIDIA CUDA backend for the graph system-scope atomic adapter.
 *
 * The ordered transport uses `::cuda::atomic_ref<T, ::cuda::thread_scope_system>`
 * from libcu++ on host-mapped memory. The libcu++ reference carries the
 * documented system-scope memory model contract for the requested order.
 *
 * `nativeSystemFetchAdd()` is the raw `::atomicAdd_system` path. CUDA does not
 * expose a reliable "system RMW on host memory" attribute (the A30 probe
 * measured a true atomic while `cudaDevAttrHostNativeAtomicSupported` read 0),
 * so that path is opt-in and must only be used after
 * `graph::verifyMappedSystemAtomics()` has confirmed it on the live allocation.
 * It is measured behaviour, not a portability contract.
 */
namespace graph::detail::cuda
{
#if GRAPH_HAS_CUDA
    namespace detail
    {
        template<typename T>
        constexpr void checkSupported()
        {
            static_assert(std::is_trivially_copyable_v<T>, "graph: atomic payloads must be trivially copyable");
            static_assert(
                ::cuda::atomic_ref<T, ::cuda::thread_scope_system>::is_always_lock_free,
                "graph: the CUDA system-scope atomic_ref is not lock-free for this type");
            static_assert(
                alignof(T) >= ::cuda::atomic_ref<T, ::cuda::thread_scope_system>::required_alignment,
                "graph: the element alignment is lower than the system atomic requires; over-align the allocation");
        }

        template<typename T>
        inline constexpr bool hasNativeSystemAdd
            = std::is_same_v<T, unsigned int> || std::is_same_v<T, int> || std::is_same_v<T, float>
              || std::is_same_v<T, double> || (std::is_integral_v<T> && sizeof(T) == 8u && std::is_unsigned_v<T>);
    } // namespace detail

    /** Atomically add @p value and return the previous value.
     *
     * Ordering contract: relaxed system-scope read-modify-write through the
     * libcu++ system-scope reference. This matches the raw `atomicAdd_system`
     * ordering and adds no release/acquire publication ordering.
     */
    template<typename T>
    ALPAKA_FN_ACC T systemFetchAdd(T* const ptr, T const value)
    {
        detail::checkSupported<T>();
        ::cuda::atomic_ref<T, ::cuda::thread_scope_system> ref(*ptr);
        return ref.fetch_add(value, ::cuda::std::memory_order_relaxed);
    }

    /** Raw `::atomicAdd_system` read-modify-write. Opt-in only.
     *
     * Ordering contract: relaxed system-scope read-modify-write. The intrinsic
     * provides no implicit publication ordering.
     *
     * Does NOT guarantee: that a mapped allocation supports it. Verify the
     * allocation first with `graph::verifyMappedSystemAtomics()`.
     */
    template<typename T>
    ALPAKA_FN_ACC T nativeSystemFetchAdd(T* const ptr, T const value)
    {
        static_assert(detail::hasNativeSystemAdd<T>, "graph: no native ::atomicAdd_system overload for this type");
        static_assert(alignof(T) >= sizeof(T), "graph: native system atomics require natural alignment");
#    if defined(__CUDA_ARCH__)
        if constexpr(std::is_same_v<T, unsigned int>)
            return ::atomicAdd_system(ptr, value);
        else if constexpr(std::is_same_v<T, int>)
            return ::atomicAdd_system(ptr, value);
        else if constexpr(std::is_same_v<T, float>)
            return ::atomicAdd_system(ptr, value);
        else if constexpr(std::is_same_v<T, double>)
            return ::atomicAdd_system(ptr, value);
        else
            return static_cast<T>(::atomicAdd_system(
                reinterpret_cast<unsigned long long*>(ptr),
                static_cast<unsigned long long>(value)));
#    else
        // The raw intrinsic is device-only. Host compilation of this
        // __host__ __device__ helper is only used for type checking; it falls
        // back to the contract-supported libcu++ system RMW.
        ::cuda::atomic_ref<T, ::cuda::thread_scope_system> ref(*ptr);
        return ref.fetch_add(value, ::cuda::std::memory_order_relaxed);
#    endif
    }

    /** Load @p ptr with system-scope acquire ordering. */
    template<typename T>
    ALPAKA_FN_ACC T systemLoadAcquire(T const* const ptr)
    {
        detail::checkSupported<T>();
        ::cuda::atomic_ref<T, ::cuda::thread_scope_system> ref(*const_cast<T*>(ptr));
        return ref.load(::cuda::std::memory_order_acquire);
    }

    /** Store @p value to @p ptr with system-scope release ordering. */
    template<typename T>
    ALPAKA_FN_ACC void systemStoreRelease(T* const ptr, T const value)
    {
        detail::checkSupported<T>();
        ::cuda::atomic_ref<T, ::cuda::thread_scope_system> ref(*ptr);
        ref.store(value, ::cuda::std::memory_order_release);
    }

    /** System-scope acquire fence. */
    ALPAKA_FN_ACC inline void systemFenceAcquire()
    {
        ::cuda::atomic_thread_fence(::cuda::std::memory_order_acquire, ::cuda::thread_scope_system);
    }

    /** System-scope release fence. */
    ALPAKA_FN_ACC inline void systemFenceRelease()
    {
        ::cuda::atomic_thread_fence(::cuda::std::memory_order_release, ::cuda::thread_scope_system);
    }
#else
    // The CUDA backend is not compiled into this build. The stubs keep the
    // backend dispatch well-formed; touching the CUDA branch fails with a
    // clear diagnostic instead of silently selecting another backend.
    namespace detail
    {
        template<typename... T>
        inline constexpr bool dependentFalse = false;
    } // namespace detail

    template<typename T>
    T systemFetchAdd(T*, T)
    {
        static_assert(detail::dependentFalse<T>, "graph: the CUDA backend is not enabled in this build");
        return T{};
    }

    template<typename T>
    T nativeSystemFetchAdd(T*, T)
    {
        static_assert(detail::dependentFalse<T>, "graph: the CUDA backend is not enabled in this build");
        return T{};
    }

    template<typename T>
    T systemLoadAcquire(T const*)
    {
        static_assert(detail::dependentFalse<T>, "graph: the CUDA backend is not enabled in this build");
        return T{};
    }

    template<typename T>
    void systemStoreRelease(T*, T)
    {
        static_assert(detail::dependentFalse<T>, "graph: the CUDA backend is not enabled in this build");
    }

    template<typename T = void>
    void systemFenceAcquire()
    {
        static_assert(detail::dependentFalse<T>, "graph: the CUDA backend is not enabled in this build");
    }

    template<typename T = void>
    void systemFenceRelease()
    {
        static_assert(detail::dependentFalse<T>, "graph: the CUDA backend is not enabled in this build");
    }
#endif
} // namespace graph::detail::cuda

#if GRAPH_HAS_CUDA
namespace graph::host
{
    /** Host-side acquire load for a libcu++ `::cuda::atomic_ref`.
     *
     * Use this overload for host-mapped memory. The matching device-side
     * primitive is `graph::systemLoadAcquire`.
     */
    template<typename T, cuda::thread_scope T_Scope>
    ALPAKA_FN_HOST T hostLoadAcquire(::cuda::atomic_ref<T, T_Scope> const& ref) noexcept
    {
        return ref.load(::cuda::std::memory_order_acquire);
    }

    /** Host-side release store for a libcu++ `::cuda::atomic_ref`. */
    template<typename T, cuda::thread_scope T_Scope>
    ALPAKA_FN_HOST void hostStoreRelease(::cuda::atomic_ref<T, T_Scope>& ref, T const value) noexcept
    {
        ref.store(value, ::cuda::std::memory_order_release);
    }
} // namespace graph::host
#endif
