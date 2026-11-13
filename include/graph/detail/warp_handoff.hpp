// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/barrier_diagnostics.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Block-shared handoff of one warp-uniform value.
 *
 * # Why this file exists
 *
 * The pinned alpaka maps `WarpGroup::broadcast` to
 * `__shfl_sync(__activemask(), ...)`. `__activemask()` only reports the lanes
 * that are *currently converged* at the call site; it does not reconverge the
 * warp. After a leader-only region that the compiler does not bracket with a
 * reconvergence point (measured: the CAS retry loop of `PacketPool::popSlot`),
 * the leader and the remaining lanes stay in two convergence groups. Every
 * later shuffle then silently returns the caller's own value on the rival
 * group, so statuses and handles stop propagating. A warp vote is never a
 * barrier, and neither is a shuffle.
 *
 * The pool therefore distributes its claim and control values through shared
 * memory and a real barrier. Shuffles remain in the vocabulary for the parts
 * that never cross a leader-only region, but no pool decision depends on them.
 *
 * # Handoff contract
 *
 * - **Who writes**: exactly one lane, `WarpGroup::isLeader()` (lane 0). The
 *   value is written to one block-shared slot. Two different handoffs must use
 *   different `T_uniqueId` values (and different value types get distinct
 *   slots); reusing an id aliases the same storage.
 * - **Ordering**: the writer publishes first, then every lane calls `sync()`,
 *   which is `onAcc::syncBlockThreads`. That barrier orders the writer's
 *   block-shared store (and the device-scope reads that produced the value)
 *   before any lane's `get()`. The writer must not hold a lock across
 *   `sync()`, and `sync()` is never executed from divergent control flow: every
 *   lane of the block calls it on the same line.
 * - **Who reads**: every lane of the block, after `sync()`, through `get()`.
 *
 * `syncBlockThreads` is the primitive, reached through `WarpBarrier`: the slot
 * is *block* shared memory, so an ordering primitive that covers the whole
 * block is exactly what the storage needs. The pool runs one warp per block,
 * where a block barrier is a warp barrier (see `warp_barrier.hpp`). Routing the
 * sync through the barrier also subjects the handoff's call site to the
 * barrier divergence diagnostic, so a handoff reached divergent is named the
 * same way a `WarpBarrier` call site is.
 *
 * Width-one accelerators (the host backend) keep the value in the handoff
 * object: the single lane is the leader, and the broadcast is the identity.
 * No shared storage is declared and no barrier is executed there.
 */
namespace graph
{
    /** Shared-memory handoff of one warp-uniform value.
     *
     * @tparam T_Acc accelerator type of the calling kernel
     * @tparam T_Value trivially copyable value type
     * @tparam T_uniqueId slot identity inside one kernel; distinct handoffs
     *         must not share an id
     */
    template<typename T_Acc, typename T_Value, std::size_t T_uniqueId>
    class WarpHandoff
    {
    public:
        using Acc = std::remove_cvref_t<T_Acc>;
        using Value = T_Value;

        static_assert(std::is_trivially_copyable_v<Value>);
        static_assert(std::is_standard_layout_v<Value>);

        static constexpr std::uint32_t compileTimeWarpSize = WarpGroup<Acc>::compileTimeWarpSize;

        ALPAKA_FN_ACC constexpr WarpHandoff() noexcept = default;

        ALPAKA_FN_ACC constexpr explicit WarpHandoff(WarpBarrierDiagnostics<Acc> const& diagnostics) noexcept
            : m_diagnostics(diagnostics)
        {
        }

        /** Publish `value` for the block. Only lane 0 writes. */
        ALPAKA_FN_ACC constexpr void publish(Acc const& acc, WarpGroup<Acc> const& warp, Value const& value) noexcept
        {
            if constexpr(compileTimeWarpSize == 1u)
            {
                (void) acc;
                (void) warp;
                m_local = value;
            }
            else
            {
                if(warp.isLeader())
                    sharedSlot(acc) = value;
            }
        }

        /** Order the writer's publication against every lane's `get()`.
         *
         * Every lane of the block must call this on the same line. The block
         * width equals the warp width in the sanctioned layout, so it is
         * passed to the barrier, which also runs the divergence diagnostic.
         * `barrierId` identifies the handoff call site; it derives from the
         * slot id so two distinct handoffs report distinct ids.
         */
        ALPAKA_FN_ACC constexpr void sync(Acc const& acc, std::uint32_t barrierId) const noexcept
        {
            if constexpr(compileTimeWarpSize == 1u)
            {
                (void) acc;
                (void) barrierId;
            }
            else
            {
                auto const blockThreads = static_cast<std::uint32_t>(
                    acc.getExtentsOf(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads)[0]);
                WarpBarrier<Acc> const barrier{m_diagnostics};
                barrier.barrier(acc, blockThreads, WarpGroup<Acc>{acc}, barrierId, ~std::uint32_t{0u}, 0u);
            }
        }

        /** Read the published value. Valid after `sync()`. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr Value get(Acc const& acc) const noexcept
        {
            if constexpr(compileTimeWarpSize == 1u)
            {
                (void) acc;
                return m_local;
            }
            else
            {
                return sharedSlot(acc);
            }
        }

    private:
        /** The one block-shared slot of this handoff.
         *
         * `declareSharedVar` is the pinned alpaka's supported static shared
         * memory mechanism; the symbol is instantiated per
         * (value type, unique id, accelerator) and lives for the kernel. */
        [[nodiscard]] ALPAKA_FN_ACC static Value& sharedSlot(Acc const& acc) noexcept
        {
            return alpaka::onAcc::declareSharedVar<Value, T_uniqueId>(acc);
        }

        /** Width-one fallback: the single lane is the leader. */
        Value m_local{};
        WarpBarrierDiagnostics<Acc> m_diagnostics{};
    };

    /** Slot ids of the pool handoffs.
     *
     * One id per logical handoff. Adding one here keeps the ids of the pool in
     * one place so a new handoff cannot silently alias an existing slot. */
    namespace handoff_id
    {
        inline constexpr std::size_t readyRingClaim = 0x5250'0001u;
        inline constexpr std::size_t readyRingPush = 0x5250'0002u;
        inline constexpr std::size_t packetPoolClaim = 0x5250'0003u;
        inline constexpr std::size_t packetPoolConfigure = 0x5250'0004u;
        inline constexpr std::size_t packetPoolRelease = 0x5250'0005u;
        inline constexpr std::size_t schedulerStartup = 0x5250'0006u;
        inline constexpr std::size_t schedulerImport = 0x5250'0007u;
    } // namespace handoff_id
} // namespace graph
