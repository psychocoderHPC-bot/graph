// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/config.hpp>

#include <atomic>
#include <cstdint>

/** Ordered device-scope loads and stores.
 *
 * These are the only primitives the pool uses to move a value from one block to
 * another. A `volatile` load is not enough on the target GPU: the record, the
 * free-list successor, and the claim words are written by another block's
 * multiplexer (a different SM), and a weak load can return a line this block
 * already cached. An acquire load (`ld.acquire.gpu` / `atomic_ref::load`) is
 * the synchronizing read that observes the release store or fence that
 * published the value. `loadRelaxed` is for counting only.
 *
 * The header is intentionally dependency-free: `packet_pool.hpp` and
 * `pool_control.hpp` both need it, and `pool_control.hpp` includes the pool.
 */
namespace graph
{
    namespace detail
    {
        namespace dev
        {
#if GRAPH_HAS_CUDA
            template<typename T>
            ALPAKA_FN_ACC inline T loadRelaxed(T const* pointer) noexcept
            {
                ::cuda::atomic_ref<T, ::cuda::thread_scope_device> ref(*const_cast<T*>(pointer));
                return ref.load(::cuda::std::memory_order_relaxed);
            }

            template<typename T>
            ALPAKA_FN_ACC inline T loadAcquire(T const* pointer) noexcept
            {
                ::cuda::atomic_ref<T, ::cuda::thread_scope_device> ref(*const_cast<T*>(pointer));
                return ref.load(::cuda::std::memory_order_acquire);
            }

            template<typename T>
            ALPAKA_FN_ACC inline void storeRelease(T* pointer, T value) noexcept
            {
                ::cuda::atomic_ref<T, ::cuda::thread_scope_device> ref(*pointer);
                ref.store(value, ::cuda::std::memory_order_release);
            }
#else
            template<typename T>
            ALPAKA_FN_ACC inline T loadRelaxed(T const* pointer) noexcept
            {
                std::atomic_ref<T> ref(*const_cast<T*>(pointer));
                return ref.load(std::memory_order_relaxed);
            }

            template<typename T>
            ALPAKA_FN_ACC inline T loadAcquire(T const* pointer) noexcept
            {
                std::atomic_ref<T> ref(*const_cast<T*>(pointer));
                return ref.load(std::memory_order_acquire);
            }

            template<typename T>
            ALPAKA_FN_ACC inline void storeRelease(T* pointer, T value) noexcept
            {
                std::atomic_ref<T> ref(*pointer);
                ref.store(value, std::memory_order_release);
            }
#endif
        } // namespace dev

        /** Device-scope relaxed load of a scheduler counter. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC inline T loadDeviceRelaxed(T_Acc const& acc, T const* pointer) noexcept
        {
            (void) acc;
            return dev::loadRelaxed(pointer);
        }

        /** Device-scope acquire load. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC inline T loadDeviceAcquire(T_Acc const& acc, T const* pointer) noexcept
        {
            (void) acc;
            return dev::loadAcquire(pointer);
        }

        /** Device-scope release store. */
        template<typename T_Acc, typename T>
        ALPAKA_FN_ACC inline void storeDeviceRelease(T_Acc const& acc, T* pointer, T value) noexcept
        {
            (void) acc;
            dev::storeRelease(pointer, value);
        }
    } // namespace detail
} // namespace graph
