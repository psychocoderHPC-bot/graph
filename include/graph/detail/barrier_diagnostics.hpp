// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/config.hpp>
#include <graph/detail/warp_group.hpp>

#include <cstdint>
#include <type_traits>

/** Divergence detection for the warp barrier.
 *
 * A `WarpBarrier` is only a valid barrier when every active lane of the warp
 * reaches the same call. With one warp per block a divergent arrival at
 * `syncBlockThreads` is a deadlock, not an error: the missing lanes never
 * arrive and the kernel hangs. This header turns that hang into a recorded
 * diagnostic so the failing call site can be named.
 *
 * # Compile-time gate
 *
 * `GRAPH_ENABLE_BARRIER_DIVERGENCE_DIAGNOSTICS` (default on) compiles the
 * detection in. A build that defines it to 0 removes the check entirely: no
 * mask query, no counter, no skip decision. A validated run is a run with the
 * diagnostics *enabled* and a divergence count of zero; a disabled build must
 * not be treated as evidence that no divergence exists.
 *
 * # Runtime gate
 *
 * `WarpBarrierDiagnostics::enabled` is the runtime switch (a `PoolConfig` flag
 * feeds it). When off, `observe()` returns a full active mask and performs no
 * device store: the production path is byte-for-byte the plain barrier. Tests
 * enable it, which costs one warp-wide mask query per barrier call.
 *
 * # What a divergence is
 *
 * Immediately before the barrier, the active-lane mask of the warp is
 * compared to the full mask of the warp width. A converged warp of width `w`
 * sees `w - 1`; anything else is divergent control flow. The diagnostic is
 * warp-scoped: only the active lanes of the divergent group run it, which is
 * exactly the group that is missing lanes, so the record lands even though the
 * barrier itself never completes.
 *
 * # Skip
 *
 * `WarpBarrierDiagnostics::skipOnDivergence` (diagnostic-only, default on for
 * tests) skips the barrier when a divergence was just recorded and the
 * diagnostics are enabled. Documented contract: a skipped barrier orders
 * nothing. The skip exists so the test fails loudly with the recorded call
 * site instead of hanging for the 120 s wait deadline; it is never used in a
 * validated run. The skip is the only behavioral difference the diagnostics
 * introduce, and it is only reachable when a divergence has already been
 * recorded, so a clean run executes the barrier unmodified.
 */
namespace graph
{
    /** Device state of the barrier divergence diagnostics.
     *
     * One instance per kernel launch, owned by the pool and zeroed before the
     * launch. `poolMaxWorkers` bounds every per-worker array: a launch beyond
     * that capacity is refused earlier, so no index can be formed out of
     * range.
     */
    struct alignas(128) BarrierDiagnosticsState
    {
        /** Total number of divergent barrier arrivals recorded. */
        alignas(128) std::uint64_t divergenceCount{0u};
        /** Barrier call-site id of the first recorded divergence. */
        alignas(128) std::uint32_t firstBarrierId{~std::uint32_t{0u}};
        /** Worker (block) index of the first recorded divergence. */
        alignas(128) std::uint32_t firstWorker{~std::uint32_t{0u}};
        /** Scheduler loop stage of the first recorded divergence. */
        alignas(128) std::uint32_t firstStage{~std::uint32_t{0u}};
        /** Scheduling-loop iteration of the first recorded divergence. */
        alignas(128) std::uint64_t firstIteration{0u};
        /** Active-lane mask observed at the first recorded divergence. */
        alignas(128) std::uint32_t firstMask{0u};
        /** Warp width at the first recorded divergence. */
        alignas(128) std::uint32_t firstWidth{0u};
        /** Divergent arrivals of each worker (per-worker histogram). */
        alignas(128) std::uint32_t perWorkerDivergences[poolMaxWorkers]{};
    };

    static_assert(std::is_trivially_copyable_v<BarrierDiagnosticsState>);
    static_assert(std::is_standard_layout_v<BarrierDiagnosticsState>);
    static_assert(alignof(BarrierDiagnosticsState) == 128u);

    /** Runtime configuration of the barrier divergence diagnostics.
     *
     * Cheap to pass by value; every worker holds one copy. `enabled` mirrors
     * the host-side flag the pool publishes; `skipOnDivergence` selects the
     * diagnostic skip behavior documented in this file.
     */
    struct BarrierDiagnosticsConfig
    {
        /** Detect divergent barrier arrivals (device store per detection). */
        bool enabled{false};
        /**
         * Skip the barrier on a detected divergence. Diagnostic-only: a
         * skipped barrier orders no memory. Never used in a validated run.
         */
        bool skipOnDivergence{false};
    };

    template<typename T_Acc>
    class WarpBarrierDiagnostics
    {
    public:
        using Acc = std::remove_cvref_t<T_Acc>;

        /** Observe the active mask just before a barrier.
         *
         * Returns the mask a converged warp of the given width must show:
         * `width - 1`. When the observed active mask differs, the divergence
         * is recorded into `m_state` and `true` is returned. A record is the
         * first divergence of this warp; later ones only increment counters.
         *
         * The record is made by the lane with the lowest active lane index
         * (the first active lane), so one warp contributes at most one record
         * per divergent group even though every active lane sees the same
         * mask. The stores are relaxed: they carry no synchronization
         * meaning.
         *
         * @param acc accelerator
         * @param warp the warp that is about to barrier
         * @param width the runtime warp width (full-mask source)
         * @param barrierId call-site identity
         * @param worker block index of the calling warp
         * @param stage scheduler loop stage (see `WorkerStage` in the
         *        scheduler), or `~0u` when unknown
         * @param iteration scheduling-loop iteration, or zero when unknown
         */
        [[nodiscard]] ALPAKA_FN_ACC bool observe(
            Acc const& acc,
            WarpGroup<Acc> const& warp,
            std::uint32_t width,
            std::uint32_t barrierId,
            std::uint32_t worker,
            std::uint32_t stage,
            std::uint64_t iteration) const noexcept
        {
            if constexpr(!GRAPH_ENABLE_BARRIER_DIVERGENCE_DIAGNOSTICS)
            {
                (void) acc;
                (void) warp;
                (void) width;
                (void) barrierId;
                (void) worker;
                (void) stage;
                (void) iteration;
                return false;
            }
            else
            {
                if(!m_config.enabled || width <= 1u)
                    return false;
                auto const fullMask = static_cast<WarpGroup<Acc>::BallotMask>((std::uint64_t{1u} << width) - 1u);
                auto const mask = warp.activeMask();
                if(mask == fullMask)
                    return false;
                // One record per divergent group: the lowest active lane index
                // writes the named fields, every active lane counts the
                // arrival. The lower bits of the mask are exactly the active
                // lanes below this one, so "none set" names the lowest active
                // lane of the group.
                auto const lane = warp.laneIndex();
                using Mask = WarpGroup<Acc>::BallotMask;
                bool const firstActive = lane < width
                                         && (mask & (Mask{1u} << lane)) != 0u
                                         && (mask & ((Mask{1u} << lane) - Mask{1u})) == 0u;
                if(worker < poolMaxWorkers)
                {
                    alpaka::onAcc::atomicAdd(
                        acc,
                        &m_state->perWorkerDivergences[worker],
                        std::uint32_t{1u},
                        alpaka::onAcc::scope::device);
                }
                (void) alpaka::onAcc::atomicAdd(
                    acc,
                    &m_state->divergenceCount,
                    std::uint64_t{1u},
                    alpaka::onAcc::scope::device);
                if(firstActive)
                {
                    (void) alpaka::onAcc::atomicExch(
                        acc,
                        &m_state->firstBarrierId,
                        barrierId,
                        alpaka::onAcc::scope::device);
                    (void) alpaka::onAcc::atomicExch(
                        acc,
                        &m_state->firstWorker,
                        worker,
                        alpaka::onAcc::scope::device);
                    (void) alpaka::onAcc::atomicExch(
                        acc,
                        &m_state->firstStage,
                        stage,
                        alpaka::onAcc::scope::device);
                    (void) alpaka::onAcc::atomicExch(
                        acc,
                        &m_state->firstIteration,
                        iteration,
                        alpaka::onAcc::scope::device);
                    (void) alpaka::onAcc::atomicExch(
                        acc,
                        &m_state->firstMask,
                        static_cast<std::uint32_t>(mask),
                        alpaka::onAcc::scope::device);
                    (void) alpaka::onAcc::atomicExch(
                        acc,
                        &m_state->firstWidth,
                        width,
                        alpaka::onAcc::scope::device);
                }
                return true;
            }
        }

    private:
        BarrierDiagnosticsState* m_state{nullptr};
        BarrierDiagnosticsConfig m_config{};

    public:
        /** @name Factory
         *  @{ */
        ALPAKA_FN_HOST_ACC constexpr static WarpBarrierDiagnostics make(
            BarrierDiagnosticsState* state,
            BarrierDiagnosticsConfig const& config) noexcept
        {
            WarpBarrierDiagnostics diagnostics{};
            diagnostics.m_state = state;
            diagnostics.m_config = config;
            return diagnostics;
        }

        /** True when the skip-on-divergence behavior is active. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr bool skipOnDivergence() const noexcept
        {
            return m_config.skipOnDivergence;
        }

        /** True when the runtime detection is enabled. */
        [[nodiscard]] ALPAKA_FN_ACC constexpr bool enabled() const noexcept
        {
            return m_config.enabled;
        }
        /** @} */
    };
} // namespace graph
