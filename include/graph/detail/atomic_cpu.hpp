// SPDX-FileCopyrightText: 2025 René Widera
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/core/common.hpp>

#include <atomic>
#include <concepts>
#include <type_traits>

/** @file
 *
 * CPU backend for the graph system-scope atomic adapter.
 *
 * A CPU "device" is the host itself, therefore the host C++ memory model is the
 * system memory model. All operations use `std::atomic_ref` with the memory
 * order that matches the primitive name. `std::atomic_ref` requires the address
 * to be aligned to `std::atomic_ref<T>::required_alignment`; every function
 * below documents that requirement and asserts that the type is always
 * lock-free for the requested operation.
 */
namespace graph::detail::cpu
{
    namespace detail
    {
        template<typename T>
        constexpr void checkSupported()
        {
            static_assert(std::is_trivially_copyable_v<T>, "graph: atomic payloads must be trivially copyable");
            static_assert(
                std::atomic_ref<T>::is_always_lock_free,
                "graph: this CPU cannot perform the requested operation lock-free; use a supported scalar type");
            static_assert(
                alignof(T) >= std::atomic_ref<T>::required_alignment,
                "graph: the element alignment is lower than std::atomic_ref requires; over-align the allocation");
        }
    } // namespace detail

    /** Atomically add @p value and return the previous value.
     *
     * Ordering contract: relaxed read-modify-write. This matches the NVIDIA
     * `atomicAdd_system` contract and adds no release/acquire publication
     * ordering. A caller that publishes data with the returned value must use an
     * explicit release operation/fence.
     *
     * Does NOT guarantee: ordering of surrounding memory accesses, and no
     * ordering for a consumer that only performs the matching acquire later.
     */
    template<typename T>
    ALPAKA_FN_HOST T systemFetchAdd(T* const ptr, T const value)
    {
        detail::checkSupported<T>();
        std::atomic_ref<T> ref(*ptr);
        return ref.fetch_add(value, std::memory_order_relaxed);
    }

    /** Load @p ptr with acquire ordering.
     *
     * Typical use: observe a producer's release store, then read the payload
     * written before that store.
     *
     * Does NOT guarantee: anything about the payload written after the release
     * operation, and no atomicity for a non-atomic pointer value.
     */
    template<typename T>
    ALPAKA_FN_HOST T systemLoadAcquire(T const* const ptr)
    {
        detail::checkSupported<T>();
        std::atomic_ref<T> ref(*const_cast<T*>(ptr));
        return ref.load(std::memory_order_acquire);
    }

    /** Store @p value to @p ptr with release ordering.
     *
     * Every write sequenced before this call becomes visible to a thread that
     * later performs an acquire load of the same address.
     *
     * Does NOT guarantee: ordering of writes performed after the release store,
     * and no ordering without a matching acquire operation.
     */
    template<typename T>
    ALPAKA_FN_HOST void systemStoreRelease(T* const ptr, T const value)
    {
        detail::checkSupported<T>();
        std::atomic_ref<T> ref(*ptr);
        ref.store(value, std::memory_order_release);
    }

    /** Acquire fence for the CPU system scope.
     *
     * Pairs with a preceding acquire load (or a load that was performed with a
     * relaxed/stronger order). All reads after the fence observe writes that
     * were released before the observed value became visible.
     */
    ALPAKA_FN_HOST inline void systemFenceAcquire()
    {
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    /** Release fence for the CPU system scope.
     *
     * Pairs with a following release store (or a store that was performed with
     * a relaxed/stronger order). All writes before the fence are ordered before
     * the published store.
     */
    ALPAKA_FN_HOST inline void systemFenceRelease()
    {
        std::atomic_thread_fence(std::memory_order_release);
    }
} // namespace graph::detail::cpu

namespace graph::host
{
    /** Host-side acquire load for a `std::atomic_ref`.
     *
     * This is the host twin of `graph::systemLoadAcquire`. On a CPU build the
     * mapped host memory is ordinary host memory, so `std::atomic_ref` is the
     * matching host primitive.
     */
    template<typename T>
    ALPAKA_FN_HOST T hostLoadAcquire(std::atomic_ref<T> const& ref) noexcept
    {
        return ref.load(std::memory_order_acquire);
    }

    /** Host-side release store for a `std::atomic_ref`. */
    template<typename T>
    ALPAKA_FN_HOST void hostStoreRelease(std::atomic_ref<T>& ref, T const value) noexcept
    {
        ref.store(value, std::memory_order_release);
    }

    /** Host-side acquire fence for CPU memory. */
    ALPAKA_FN_HOST inline void hostFenceAcquire() noexcept
    {
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    /** Host-side release fence for CPU memory. */
    ALPAKA_FN_HOST inline void hostFenceRelease() noexcept
    {
        std::atomic_thread_fence(std::memory_order_release);
    }
} // namespace graph::host
