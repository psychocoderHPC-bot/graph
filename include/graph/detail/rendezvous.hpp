// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/config.hpp>
#include <graph/detail/warp_group.hpp>

#if GRAPH_HAS_CUDA
#    include <cuda_runtime.h>

#    include <cuda/atomic>
#    include <cuda/std/atomic>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <type_traits>

/** @file
 *
 * Startup rendezvous kernel for the persistent worker pool.
 *
 * The kernel is the empirical half of the co-residency proof. Every warp leader
 * arrives once, records its block id and a device clock sample, then spins until
 * the expected number of workers has arrived or a deadline expires. A worker
 * increments `completed` only when it leaves the spin because it observed the
 * full arrival count. Sequential waves cannot fake that: a block from a later
 * wave arrives after the earlier blocks gave up at the deadline, so the earlier
 * blocks never increment `completed`. The host therefore requires
 * `completed == workerCount`, not merely `arrived == workerCount`.
 *
 * The deadline bounds each worker's spin, so a non-resident grid fails the test
 * instead of hanging it.
 *
 * The same kernel text runs on the CPU executor. There it proves progress and
 * that the launch matches the CPU policy; it is not GPU residency evidence.
 *
 * A `clock64()` sample is only comparable within one SM. The per-worker samples
 * are recorded for diagnostics; the reliable wall-clock arrival time comes from
 * the host around the queue wait.
 */
namespace graph
{
    /** Layout of the rendezvous state buffer, in 64-bit words.
     *
     * Header words 0..7 are fixed. Three arrays of `capacity` words follow:
     * block ids, clock samples, and the final observed arrival count per slot.
     */
    struct RendezvousLayout
    {
        static constexpr std::uint32_t arrived = 0u;
        static constexpr std::uint32_t expected = 1u;
        static constexpr std::uint32_t launchCount = 2u;
        static constexpr std::uint32_t allArrived = 3u;
        static constexpr std::uint32_t firstClock = 4u;
        static constexpr std::uint32_t lastClock = 5u;
        static constexpr std::uint32_t completed = 6u;
        static constexpr std::uint32_t headerWords = 8u;

        ALPAKA_FN_HOST_ACC static constexpr std::uint32_t blockIdOffset(std::uint32_t capacity) noexcept
        {
            static_cast<void>(capacity);
            return headerWords;
        }

        ALPAKA_FN_HOST_ACC static constexpr std::uint32_t clockOffset(std::uint32_t capacity) noexcept
        {
            return headerWords + capacity;
        }

        ALPAKA_FN_HOST_ACC static constexpr std::uint32_t observedOffset(std::uint32_t capacity) noexcept
        {
            return headerWords + 2u * capacity;
        }

        ALPAKA_FN_HOST_ACC static constexpr std::uint32_t wordCount(std::uint32_t capacity) noexcept
        {
            return headerWords + 3u * capacity;
        }
    };

    namespace detail
    {
        template<typename T_Api>
        inline constexpr bool isCudaLaunchApi = std::is_same_v<std::remove_cvref_t<T_Api>, alpaka::api::Cuda>;

        namespace cuda
        {
#if GRAPH_HAS_CUDA
            /** Device cycle counter for the persistent kernel.
             *
             * The host pass of this `__host__ __device__` helper returns zero;
             * only the device pass reads `clock64()`.
             */
            ALPAKA_FN_ACC inline std::uint64_t cycleClock() noexcept
            {
#    if defined(__CUDA_ARCH__)
                return static_cast<std::uint64_t>(clock64());
#    else
                return 0u;
#    endif
            }

            /** Device-scope relaxed load.
             *
             * A relaxed load is enough for polling the arrival counter and it
             * avoids turning the spin into a contended read-modify-write.
             */
            ALPAKA_FN_ACC inline std::uint64_t atomicLoadRelaxed(std::uint64_t const* ptr) noexcept
            {
                ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_device> ref(*const_cast<std::uint64_t*>(ptr));
                return ref.load(::cuda::std::memory_order_relaxed);
            }
#else
            inline std::uint64_t cycleClock() noexcept
            {
                return 0u;
            }

            inline std::uint64_t atomicLoadRelaxed(std::uint64_t const* ptr) noexcept
            {
                return *ptr;
            }
#endif
        } // namespace cuda

        /** Cycle sample on the accelerator, zero on the host backend.
         *
         * Backend selection uses the alpaka API tag; the CPU backend has no
         * cycle counter, so its deadline is an iteration budget instead.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline std::uint64_t cycleNow(T_Acc const& acc) noexcept
        {
            using Api = ALPAKA_TYPEOF(acc.getApi());
            if constexpr(isCudaLaunchApi<Api>)
                return cuda::cycleClock();
            else
            {
                static_cast<void>(acc);
                return 0u;
            }
        }

        /** Relaxed load from a device-visible counter on any backend. */
        template<typename T_Acc>
        ALPAKA_FN_ACC inline std::uint64_t loadArrivalCount(T_Acc const& acc, std::uint64_t const* ptr) noexcept
        {
            using Api = ALPAKA_TYPEOF(acc.getApi());
            if constexpr(isCudaLaunchApi<Api>)
                return cuda::atomicLoadRelaxed(ptr);
            else
            {
                static_cast<void>(acc);
                std::atomic_ref<std::uint64_t> ref(*const_cast<std::uint64_t*>(ptr));
                return ref.load(std::memory_order_relaxed);
            }
        }
    } // namespace detail

    /** One-warp-per-block (or N-warp-per-block) startup rendezvous.
     *
     * @param state rendezvous state buffer, see `RendezvousLayout`
     * @param capacity number of per-worker recording slots (>= expected workers)
     * @param deadline CUDA: maximum cycles each leader may spin; CPU: maximum
     *        spin iterations. Passing the deadline makes the worker leave the
     *        rendezvous without incrementing `completed`; the host reports that.
     */
    struct RendezvousKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            std::uint64_t* state,
            std::uint32_t capacity,
            std::uint64_t deadline) const
        {
            if(capacity == 0u)
                return;

            graph::WarpGroup<T_Acc> const warp{acc};
            if(!warp.isLeader())
                return;

            auto const blockIndex = static_cast<std::uint64_t>(acc[alpaka::layer::block].idx()[0]);
            std::uint64_t const start = graph::detail::cycleNow(acc);
            std::uint64_t const expectedCount = state[RendezvousLayout::expected];

            // Exactly one launch marker per kernel launch: block zero, first warp.
            if(blockIndex == 0u && alpaka::onAcc::warp::getWarpIdx(acc) == 0u)
                alpaka::onAcc::atomicAdd(
                    acc,
                    state + RendezvousLayout::launchCount,
                    std::uint64_t{1},
                    alpaka::onAcc::scope::device);

            std::uint64_t const slot = alpaka::onAcc::atomicAdd(
                acc,
                state + RendezvousLayout::arrived,
                std::uint64_t{1},
                alpaka::onAcc::scope::device);
            auto const index
                = static_cast<std::uint32_t>(std::min<std::uint64_t>(slot, static_cast<std::uint64_t>(capacity) - 1u));

            if(slot < static_cast<std::uint64_t>(capacity))
            {
                state[RendezvousLayout::blockIdOffset(capacity) + index] = blockIndex;
                state[RendezvousLayout::clockOffset(capacity) + index] = start;
            }
            if(slot == 0u)
                state[RendezvousLayout::firstClock] = start;
            if(slot + 1u == expectedCount)
            {
                state[RendezvousLayout::lastClock] = graph::detail::cycleNow(acc);
                alpaka::onAcc::atomicExch(
                    acc,
                    state + RendezvousLayout::allArrived,
                    std::uint64_t{1},
                    alpaka::onAcc::scope::device);
            }

            bool sawAll = false;
            std::uint64_t observed = 0u;
            std::uint64_t iterations = 0u;
            while(true)
            {
                observed = graph::detail::loadArrivalCount(acc, state + RendezvousLayout::arrived);
                if(observed >= expectedCount)
                {
                    sawAll = true;
                    break;
                }

                ++iterations;
                using Api = ALPAKA_TYPEOF(acc.getApi());
                if constexpr(graph::detail::isCudaLaunchApi<Api>)
                {
                    if(graph::detail::cycleNow(acc) - start > deadline)
                        break;
                }
                else
                {
                    if(iterations > deadline)
                        break;
                }
            }

            state[RendezvousLayout::observedOffset(capacity) + index] = observed;
            if(sawAll)
                alpaka::onAcc::atomicAdd(
                    acc,
                    state + RendezvousLayout::completed,
                    std::uint64_t{1},
                    alpaka::onAcc::scope::device);
        }
    };
} // namespace graph
