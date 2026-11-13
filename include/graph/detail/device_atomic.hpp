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
 *
 * # Project rule: cross-worker loop exit conditions
 *
 * A flag that one worker writes and another reads in a wait loop (a stop or
 * drain flag, a startup `initDone`, a ring or pool counter observed from another
 * block) must be observable on every iteration of the wait. The rule:
 *
 * - The flag is **atomic or volatile**, and it is **re-read inside the loop
 *   body on every iteration**. An atomic load with `memory_order_acquire`
 *   (`loadDeviceAcquire`, or `systemLoadAcquire` on a mapped word) is preferred:
 *   it pairs with the writer's release store and also orders the payload the
 *   flag publishes. A plain `volatile` read is sufficient only when the word is
 *   genuinely a single flag whose publication needs no ordering and whose writer
 *   is a different *worker* (the compiler cannot cache a `volatile` read across
 *   the loop).
 * - A **plain member must never be a loop exit condition in device code.** The
 *   compiler may prove a plain (non-atomic, non-volatile) local or member
 *   invariant inside the loop and hoist the load out of it; the worker then
 *   spins on the first value forever, even when another worker has set the flag.
 * - **Never cache a cross-worker flag in a local before the loop**
 *   (`auto stop = s.stop;` and then loop on `stop`): the copy is not
 *   observable, so the loop exits only if the flag was already set at entry.
 *
 * The regression test `tests/integration/test_hoisted_flag.cpp` pins this: a
 * worker spins on a flag set by another worker after a delay, with a budget; a
 * cached (hoisted) load exhausts the budget and fails, a re-reading load
 * finishes in well under it.
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
