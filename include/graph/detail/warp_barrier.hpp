// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/warp_group.hpp>

#include <cstdint>

/** A real, warp-scoped barrier.
 *
 * This is the one and only place in the project that orders the memory of the
 * lanes of a single worker warp. It exists because a warp *vote*
 * (`WarpGroup::all`/`any`, CUDA `__all_sync`) is not a barrier: it only
 * observes predicates, it does not order one lane's stores against another
 * lane's later reads, and it gives no convergence guarantee that a
 * correctness argument may rely on. The ReadyRing and the PacketPool publish
 * and consume through this barrier, never through a vote.
 *
 * # Policy
 *
 * - **One warp per block** (the sanctioned bring-up configuration,
 *   `blockThreads == warpSize`): `barrier()` maps to
 *   `onAcc::syncBlockThreads(acc)`, which is `__syncthreads()` on CUDA. With
 *   exactly one warp in the block, a block-wide barrier is precisely a warp
 *   barrier, and it orders every lane's stores against every lane's later
 *   reads. This is explicitly permitted for the bring-up layout.
 *
 * - **More than one warp per block**: `barrier()` is *refused*. A block-wide
 *   `__syncthreads()` would also order the other warps in the block (a
 *   different, co-residency-dependent schedule), and a bare vote would give no
 *   ordering at all, so neither is an acceptable silent fallback. The call
 *   trips a runtime flag the kernel can assert on; it does not order any
 *   memory. Multi-warp blocks therefore require a separately validated
 *   warp-local barrier (shuffle-based arrive/wait over device memory) before
 *   this header may be used with `blockThreads > warpSize`.
 *
 * The width-one (host) fallback is the identity: a single-lane warp has no
 * other lane whose stores must be ordered, so `barrier()` does nothing.
 *
 * The barrier is invoked exactly warp-uniformly: every active lane of the
 * warp reaches the same `barrier()` call. It is never held across a lock
 * acquisition, a wait, or a callback.
 */
namespace graph
{
    template<typename T_Acc>
    class WarpBarrier
    {
    public:
        using Acc = std::remove_cvref_t<T_Acc>;

        ALPAKA_FN_ACC constexpr WarpBarrier() noexcept = default;

        /** True for a block that holds exactly one warp (the bring-up layout).
         *
         * `blockThreads` is the launch's `blockDim` in threads. When it equals
         * the runtime warp width there is exactly one warp in the block and a
         * block barrier is a valid warp barrier.
         */
        [[nodiscard]] ALPAKA_FN_ACC constexpr bool isOneWarpPerBlock(Acc const& acc, std::uint32_t blockThreads)
            const noexcept
        {
            auto const width = alpaka::onAcc::warp::getSize(acc);
            return width == 1u || blockThreads == width;
        }

        /** Order the lanes of the calling warp.
         *
         * Every active lane must call this, with the same `blockThreads`, on
         * the same line. It maps to a block barrier for the one-warp-per-block
         * layout, is the identity for the width-one host fallback, and refuses
         * (tripping `m_refused`) otherwise. It must not be called while the
         * caller holds any lock or is inside a wait or callback.
         */
        ALPAKA_FN_ACC constexpr void barrier(Acc const& acc, std::uint32_t blockThreads, WarpGroup<Acc> const& warp)
            const noexcept
        {
            auto const width = alpaka::onAcc::warp::getSize(acc);
            if(width <= 1u)
            {
                // Host / single-lane warp: nothing to order.
                (void) acc;
                (void) blockThreads;
                (void) warp;
                return;
            }
            if(blockThreads == width)
            {
                // One warp per block: a block barrier is a warp barrier.
                alpaka::onAcc::syncBlockThreads(acc);
                return;
            }
            // More than one warp per block: no validated warp-local barrier is
            // available in this header. Refuse loudly instead of falling back
            // to a vote or an over-wide block barrier.
            m_refused = true;
        }

        /** True when `barrier()` ever refused a multi-warp block. Kernels must
         *  assert this is false after use; it is exposed so a test can turn the
         *  refusal into a hard failure instead of a silent mis-ordering. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr bool refused() const noexcept
        {
            return m_refused;
        }

    private:
        mutable bool m_refused = false;
    };
} // namespace graph
