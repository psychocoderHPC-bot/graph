// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>

/** One worker is one hardware warp.
 *
 * `WarpGroup` is the only place in the project that touches warp intrinsics.
 * Functors and the scheduler ask it for the runtime width, the lane index,
 * warp-uniform broadcasts, ballots, and ballot ranks. Keeping the intrinsics
 * here means the rest of the code never branches on a concrete backend.
 *
 * Width-one fallbacks: host accelerators report a warp size of one. The
 * compile-time branch below (`Acc::getWarpSize() == 1`) makes broadcast the
 * identity, ballot a one-bit mask, and rank zero. A runtime width of one is
 * handled by the same fallbacks, so a device that executes a single-lane warp
 * is covered too.
 *
 * No warp barrier is exposed here. The initial pool uses one warp per block so
 * that block-wide synchronization is sufficient; a dedicated warp barrier is a
 * later, separately validated adapter if multi-warp blocks are enabled.
 */
namespace graph
{
    template<typename T_Acc>
    class WarpGroup
    {
    public:
        using Acc = std::remove_cvref_t<T_Acc>;
        using BallotMask = std::uint64_t;

        ALPAKA_FN_ACC constexpr explicit WarpGroup(T_Acc const& acc) noexcept : m_acc(&acc)
        {
        }

        /** Compile-time warp size of the accelerator instantiation. */
        static constexpr std::uint32_t compileTimeWarpSize = Acc::getWarpSize();

        static_assert(compileTimeWarpSize >= 1u);

        /** Runtime hardware warp width. */
        ALPAKA_FN_ACC constexpr std::uint32_t workerWidth() const noexcept
        {
            return alpaka::onAcc::warp::getSize(*m_acc);
        }

        /** Lane index of the calling thread inside its warp. */
        ALPAKA_FN_ACC constexpr std::uint32_t laneIndex() const noexcept
        {
            return alpaka::onAcc::warp::getLaneIdx(*m_acc);
        }

        /** True for the first lane of the warp. */
        ALPAKA_FN_ACC constexpr bool isLeader() const noexcept
        {
            return laneIndex() == 0u;
        }

        /** Warp-uniform broadcast from `srcLane` (identity for width one). */
        template<typename T>
        ALPAKA_FN_ACC constexpr T broadcast(T const& value, std::uint32_t srcLane = 0u) const noexcept
            requires(std::is_trivially_copyable_v<T>)
        {
            if constexpr(compileTimeWarpSize == 1u)
            {
                return value;
            }
            else
            {
                auto const width = workerWidth();
                if(width <= 1u)
                    return value;
                return alpaka::onAcc::warp::shfl(*m_acc, value, srcLane, width);
            }
        }

        /** Warp ballot as a fixed-width mask (one bit for width one). */
        ALPAKA_FN_ACC constexpr BallotMask ballot(bool predicate) const noexcept
        {
            if constexpr(compileTimeWarpSize == 1u)
            {
                return predicate ? BallotMask{1u} : BallotMask{0u};
            }
            else
            {
                auto const width = workerWidth();
                if(width <= 1u)
                    return predicate ? BallotMask{1u} : BallotMask{0u};
                return static_cast<BallotMask>(alpaka::onAcc::warp::ballot(*m_acc, predicate ? 1 : 0));
            }
        }

        /** Number of set bits in a ballot mask. */
        ALPAKA_FN_ACC static constexpr std::uint32_t popcount(BallotMask mask) noexcept
        {
            return static_cast<std::uint32_t>(alpaka::popcount(mask));
        }

        /** Position of the calling lane among the set lanes of `mask`. */
        ALPAKA_FN_ACC constexpr std::uint32_t rank(BallotMask mask) const noexcept
        {
            if constexpr(compileTimeWarpSize == 1u)
            {
                return 0u;
            }
            else
            {
                auto const lane = laneIndex();
                if(lane >= 64u)
                    return popcount(mask);
                auto const lowerMask = mask & ((BallotMask{1u} << lane) - BallotMask{1u});
                return popcount(lowerMask);
            }
        }

        /** True when every active lane satisfies `predicate`. */
        ALPAKA_FN_ACC constexpr bool all(bool predicate) const noexcept
        {
            if constexpr(compileTimeWarpSize == 1u)
            {
                return predicate;
            }
            else
            {
                auto const width = workerWidth();
                if(width <= 1u)
                    return predicate;
                return alpaka::onAcc::warp::all(*m_acc, predicate ? 1 : 0);
            }
        }

        /** True when at least one active lane satisfies `predicate`. */
        ALPAKA_FN_ACC constexpr bool any(bool predicate) const noexcept
        {
            if constexpr(compileTimeWarpSize == 1u)
            {
                return predicate;
            }
            else
            {
                auto const width = workerWidth();
                if(width <= 1u)
                    return predicate;
                return alpaka::onAcc::warp::any(*m_acc, predicate ? 1 : 0);
            }
        }

    private:
        Acc const* m_acc;
    };
} // namespace graph
