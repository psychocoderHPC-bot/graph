// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/views.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/rng.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

/** Named particle-feedback functors (design section 8).
 *
 * The five structs of the semantics table, `GenerateParticles`,
 * `FilterByProperty`, `ProcessA`, `ProcessB`, and `DestroyParticles`, in one
 * header that compiles unchanged for the CPU executor and the CUDA backend.
 * There is no accelerator selection: the per-particle core is a pure function
 * of its inputs, and every `operator()` is a plain function that takes the
 * accelerator by tag. The same struct body runs on the host and on the device.
 *
 * # Particle record
 *
 * `Particle` is the element of every stream of the example. The field order
 * and widths are fixed:
 *
 * | field      | type                | meaning                                  |
 * | ---------- | ------------------- | ---------------------------------------- |
 * | `id`       | `std::uint64_t`     | unique source id                         |
 * | `property` | `std::uint32_t`     | draw at generation, preserved forever    |
 * | `weight`   | `std::uint32_t`     | accumulated weight, starts at 0          |
 * | `visit`    | `std::uint32_t`     | feedback rounds completed (starts at 0)  |
 *
 * `weight` is a fixed-width unsigned 32-bit integer. Every increment goes
 * through `checkedWeightAdd()`, which reports overflow in 64-bit arithmetic
 * instead of wrapping; the functor surfaces that through its per-call error
 * out-parameter.
 *
 * `visit` starts at 0 for the first filter pass and increments by 1 on every
 * feedback return, before the particle re-enters the filter. Draws are pure
 * functions of the key tuple, so the value of the counter only renames the
 * key; the 32-bit width is sufficient for the example's visit guard.
 *
 * # The two halves of every functor
 *
 * - `apply(...)` is the per-particle core. It is a pure function of its
 *   arguments with no I/O, so it can be tested without the scheduler, driven
 *   by a host worklist, and executed by all lanes of a device warp.
 * - `operator()` is the project `OperationFn` ABI: `void(T_Acc const&,
 *   JobDescriptor const&, PacketAccessor const&)`. It reads its input packet
 *   span, calls `apply()` per logical element, writes its output through
 *   `StreamAssembler` (reserve, write pointers, commit), and retires the input
 *   exactly once after the commit.
 *
 *   Two additional plain parameters follow the ABI parameters: the
 *   `StreamAssembler<Particle>` view of the destination streams and a `bool*`
 *   error out-parameter the worker forwards to the scheduler. Binding those
 *   per call is the job of the pool's registration wrapper (a later
 *   milestone); a functor that needs state beyond the ABI takes it that way
 *   instead of hiding it in globals.
 *
 * # Semantics (integers on the 2^32 scale, see `graph/rng.hpp`)
 *
 * | functor              | behavior                                                              |
 * | --------------------- | --------------------------------------------------------------------- |
 * | `GenerateParticles`   | id from the request, property = PROPERTY draw at visit 0, weight 0.  |
 * | `FilterByProperty`    | FILTER draw `>= property` -> Bin 1, else Bin 0.                       |
 * | `ProcessA`            | DECISION_A draw `>= 2^31`: weight += 1000, back to the filter input   |
 * |                       | with visit incremented; else weight unchanged, to the sink.           |
 * | `ProcessB`            | same rule with DECISION_B and weight += 1.                            |
 * | `DestroyParticles`    | writes the final weight to the particle's unique result slot, retires.|
 *
 * No allocation, no host calls, no block-wide barriers, and no floating point
 * in the decision path.
 */
namespace graph
{
    /** Element carried by every stream of the particle-feedback example. */
    struct Particle
    {
        std::uint64_t id{0u};
        std::uint32_t property{0u};
        std::uint32_t weight{0u};
        std::uint32_t visit{0u};
    };

    static_assert(PacketElement<Particle>);
    static_assert(std::is_standard_layout_v<Particle>);
    static_assert(sizeof(Particle) == 24u);

    namespace particle_feedback
    {
        /** One element out of the per-particle core: where the particle goes
         *  next and whether its weight arithmetic overflowed.
         *
         * `Destination` names the logical stream: `0` Bin 0, `1` Bin 1, `2`
         * the shared filter input, `3` the sink. Each `operator()` maps it
         * onto its own output ports.
         */
        enum class Destination : std::uint32_t
        {
            Bin0 = 0u,
            Bin1 = 1u,
            FilterInput = 2u,
            Sink = 3u
        };

        struct ApplyResult
        {
            Destination destination{Destination::Bin0};
            bool error{false};
        };

        /** Checked weight increment: reports overflow instead of wrapping. */
        ALPAKA_FN_HOST_ACC constexpr bool checkedWeightAdd(std::uint32_t& weight, std::uint32_t add) noexcept
        {
            std::uint64_t const sum = static_cast<std::uint64_t>(weight) + add;
            if(sum > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
                return false;
            weight = static_cast<std::uint32_t>(sum);
            return true;
        }

        /** Stream ids of the example. Mirrors `StreamIds` of
         *  `graph/particle_feedback_graph.hpp` as plain values so this header
         *  stays self-contained. */
        ALPAKA_FN_HOST_ACC constexpr StreamId streamFilterInput() noexcept
        {
            return StreamId{1u};
        }

        ALPAKA_FN_HOST_ACC constexpr StreamId streamBin0() noexcept
        {
            return StreamId{2u};
        }

        ALPAKA_FN_HOST_ACC constexpr StreamId streamBin1() noexcept
        {
            return StreamId{3u};
        }

        ALPAKA_FN_HOST_ACC constexpr StreamId streamSink() noexcept
        {
            return StreamId{4u};
        }
    } // namespace particle_feedback

    /** Source node: assigns id, property, and the initial weight 0. */
    struct GenerateParticles
    {
        using Seed = RngSeed;

        /** Per-particle core. Pure: no I/O, no shared state. */
        ALPAKA_FN_HOST_ACC static constexpr Particle apply(Seed seed, std::uint64_t id) noexcept
        {
            return Particle{id, rngDraw32(seed, id, 0u, RngPurpose::Property), 0u, 0u};
        }

        /** `OperationFn`-compatible entry point.
         *
         * No input packet. The request of the job selects the id range:
         * element `i` of the output is particle `request.firstId + i` for
         * `i < request.count`. Output ports: 0 = filter input.
         */
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            JobDescriptor const& job,
            PacketAccessor const& accessor,
            StreamAssembler<Particle> const& assembler,
            bool* error) const noexcept
        {
            (void) acc;
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const count = job.request.count;
            auto const seed = job.request.seed;
            auto const firstId = job.request.firstId;

            bool ok = count != 0u;
            if(ok)
            {
                particle_feedback::AppendRequest const request = particle_feedback::AppendRequest{
                    particle_feedback::streamFilterInput(),
                    static_cast<std::uint32_t>(count)};
                particle_feedback::StreamReservation<Particle> reservation{};
                auto const status = reservation.tryReserve(
                    acc,
                    warp,
                    warp.broadcast(static_cast<std::uint32_t>(count) + 1u),
                    assembler,
                    &request,
                    1u,
                    nullptr,
                    0u,
                    nullptr,
                    0u);
                ok = status == AppendStatus::Ok;
                if(ok)
                {
                    for(std::uint64_t i = lane; i < count; i += width)
                    {
                        auto const target = reservation.writePointer(0u, static_cast<std::uint32_t>(i));
                        if(target == nullptr)
                        {
                            ok = false;
                            break;
                        }
                        *target = apply(seed, firstId + i);
                    }
                    ok = ok && reservation.commit(acc, warp);
                    if(ok)
                        reservation.retireInput(warp);
                    else
                        reservation.cancel(acc, warp);
                }
            }
            if(!ok && error != nullptr && warp.isLeader())
                *error = true;
        }
    };

    /** Filter node: routes each particle to Bin 0 or Bin 1. */
    struct FilterByProperty
    {
        using Seed = RngSeed;

        /** Per-particle core. Pure: draws its own FILTER value. */
        ALPAKA_FN_HOST_ACC static constexpr particle_feedback::ApplyResult apply(
            Seed seed,
            Particle const& particle) noexcept
        {
            auto const draw = rngDraw32(seed, particle.id, particle.visit, RngPurpose::Filter);
            return particle_feedback::ApplyResult{
                draw >= particle.property ? particle_feedback::Destination::Bin1
                                          : particle_feedback::Destination::Bin0,
                false};
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            JobDescriptor const& job,
            PacketAccessor const& accessor,
            StreamAssembler<Particle> const& assembler,
            bool* error) const noexcept
        {
            (void) acc;
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const seed = job.request.seed;

            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            auto const count = input.size();

            // Pass 1: every lane decides its disjoint slice; the reservation
            // needs the warp-uniform per-destination total.
            std::uint32_t count0 = 0u;
            std::uint32_t count1 = 0u;
            bool ok = true;
            for(std::size_t i = lane; i < count; ++i)
            {
                auto const result = apply(seed, input.read(i));
                if(static_cast<std::uint32_t>(result.destination)
                   == static_cast<std::uint32_t>(particle_feedback::Destination::Bin1))
                {
                    ++count1;
                }
                else
                {
                    ++count0;
                }
                ok = ok && !result.error;
            }
            particle_feedback::AppendRequest requests[2u]{};
            requests[0u] = particle_feedback::AppendRequest{particle_feedback::streamBin0(), 0u};
            requests[1u] = particle_feedback::AppendRequest{particle_feedback::streamBin1(), 0u};
            for(std::uint32_t src = 0u; src < width; ++src)
            {
                requests[0u].itemCount += warp.broadcast(count0, src);
                requests[1u].itemCount += warp.broadcast(count1, src);
            }

            particle_feedback::StreamReservation<Particle> reservation{};
            auto const status = reservation.tryReserve(
                acc,
                warp,
                warp.broadcast(static_cast<std::uint32_t>(count) + 1u),
                assembler,
                requests,
                2u,
                nullptr,
                0u,
                nullptr,
                0u);
            ok = ok && status == AppendStatus::Ok;
            if(ok)
            {
                // Pass 2: write each element through its destination's rank.
                for(std::size_t i = lane; i < count; ++i)
                {
                    auto const particle = input.read(i);
                    auto const result = apply(seed, particle);
                    auto const isBin1 = static_cast<std::uint32_t>(result.destination)
                                        == static_cast<std::uint32_t>(particle_feedback::Destination::Bin1);
                    auto const destination
                        = isBin1 ? particle_feedback::streamBin1() : particle_feedback::streamBin0();
                    auto const group = assembler.destinationGroup(warp, destination, true);
                    auto const target = reservation.writePointer(isBin1 ? 1u : 0u, group.rank);
                    if(target == nullptr)
                    {
                        ok = false;
                        break;
                    }
                    *target = particle;
                }
                ok = ok && reservation.commit(acc, warp);
                if(ok)
                    reservation.retireInput(warp);
                else
                    reservation.cancel(acc, warp);
            }
            if(!ok && error != nullptr && warp.isLeader())
                *error = true;
        }
    };

    /** Bin 0 processor: +1000 on survival, back to the filter input. */
    struct ProcessA
    {
        using Seed = RngSeed;
        static constexpr std::uint32_t weightIncrement = 1000u;

        /** Per-particle core. Pure: draws its own DECISION_A value. */
        ALPAKA_FN_HOST_ACC static constexpr particle_feedback::ApplyResult apply(
            Seed seed,
            Particle const& particle) noexcept
        {
            auto const draw = rngDraw32(seed, particle.id, particle.visit, RngPurpose::DecisionA);
            if(draw >= halfScale)
            {
                auto copy = particle;
                if(!particle_feedback::checkedWeightAdd(copy.weight, weightIncrement))
                    return particle_feedback::ApplyResult{particle_feedback::Destination::Sink, true};
                ++copy.visit;
                return particle_feedback::ApplyResult{particle_feedback::Destination::FilterInput, false};
            }
            return particle_feedback::ApplyResult{particle_feedback::Destination::Sink, false};
        }

        /** The particle the feedback stream carries: weight + 1000, visit + 1.
         *  Only called on the flag-1 branch of `apply()`, so the checked add
         *  cannot fail here. */
        ALPAKA_FN_HOST_ACC static constexpr Particle feedbackParticle(Particle const& p) noexcept
        {
            auto copy = p;
            (void) particle_feedback::checkedWeightAdd(copy.weight, weightIncrement);
            ++copy.visit;
            return copy;
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            JobDescriptor const& job,
            PacketAccessor const& accessor,
            StreamAssembler<Particle> const& assembler,
            bool* error) const noexcept
        {
            (void) acc;
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const seed = job.request.seed;

            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            auto const count = input.size();

            std::uint32_t countFeedback = 0u;
            std::uint32_t countSink = 0u;
            bool ok = true;
            for(std::size_t i = lane; i < count; ++i)
            {
                auto const result = apply(seed, input.read(i));
                if(static_cast<std::uint32_t>(result.destination)
                   == static_cast<std::uint32_t>(particle_feedback::Destination::FilterInput))
                {
                    ++countFeedback;
                }
                else
                {
                    ++countSink;
                }
                ok = ok && !result.error;
            }

            particle_feedback::AppendRequest requests[2u]{};
            requests[0u] = particle_feedback::AppendRequest{particle_feedback::streamFilterInput(), 0u};
            requests[1u] = particle_feedback::AppendRequest{particle_feedback::streamSink(), 0u};
            for(std::uint32_t src = 0u; src < width; ++src)
            {
                requests[0u].itemCount += warp.broadcast(countFeedback, src);
                requests[1u].itemCount += warp.broadcast(countSink, src);
            }

            particle_feedback::StreamReservation<Particle> reservation{};
            auto const status = reservation.tryReserve(
                acc,
                warp,
                warp.broadcast(static_cast<std::uint32_t>(count) + 1u),
                assembler,
                requests,
                2u,
                nullptr,
                0u,
                nullptr,
                0u);
            ok = ok && status == AppendStatus::Ok;
            if(ok)
            {
                for(std::size_t i = lane; i < count; ++i)
                {
                    auto const particle = input.read(i);
                    auto const result = apply(seed, particle);
                    auto const toFeedback = static_cast<std::uint32_t>(result.destination)
                                            == static_cast<std::uint32_t>(particle_feedback::Destination::FilterInput);
                    auto const destination
                        = toFeedback ? particle_feedback::streamFilterInput() : particle_feedback::streamSink();
                    auto const group = assembler.destinationGroup(warp, destination, true);
                    auto const target = reservation.writePointer(toFeedback ? 0u : 1u, group.rank);
                    if(target == nullptr)
                    {
                        ok = false;
                        break;
                    }
                    *target = toFeedback ? feedbackParticle(particle) : particle;
                }
                ok = ok && reservation.commit(acc, warp);
                if(ok)
                    reservation.retireInput(warp);
                else
                    reservation.cancel(acc, warp);
            }
            if(!ok && error != nullptr && warp.isLeader())
                *error = true;
        }
    };

    /** Bin 1 processor: +1 on survival, back to the filter input. */
    struct ProcessB
    {
        using Seed = RngSeed;
        static constexpr std::uint32_t weightIncrement = 1u;

        /** Per-particle core. Pure: draws its own DECISION_B value. */
        ALPAKA_FN_HOST_ACC static constexpr particle_feedback::ApplyResult apply(
            Seed seed,
            Particle const& particle) noexcept
        {
            auto const draw = rngDraw32(seed, particle.id, particle.visit, RngPurpose::DecisionB);
            if(draw >= halfScale)
            {
                auto copy = particle;
                if(!particle_feedback::checkedWeightAdd(copy.weight, weightIncrement))
                    return particle_feedback::ApplyResult{particle_feedback::Destination::Sink, true};
                ++copy.visit;
                return particle_feedback::ApplyResult{particle_feedback::Destination::FilterInput, false};
            }
            return particle_feedback::ApplyResult{particle_feedback::Destination::Sink, false};
        }

        /** The particle the feedback stream carries: weight + 1, visit + 1.
         *  Only called on the flag-1 branch of `apply()`, so the checked add
         *  cannot fail here. */
        ALPAKA_FN_HOST_ACC static constexpr Particle feedbackParticle(Particle const& p) noexcept
        {
            auto copy = p;
            (void) particle_feedback::checkedWeightAdd(copy.weight, weightIncrement);
            ++copy.visit;
            return copy;
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            JobDescriptor const& job,
            PacketAccessor const& accessor,
            StreamAssembler<Particle> const& assembler,
            bool* error) const noexcept
        {
            (void) acc;
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const seed = job.request.seed;

            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            auto const count = input.size();

            std::uint32_t countFeedback = 0u;
            std::uint32_t countSink = 0u;
            bool ok = true;
            for(std::size_t i = lane; i < count; ++i)
            {
                auto const result = apply(seed, input.read(i));
                if(static_cast<std::uint32_t>(result.destination)
                   == static_cast<std::uint32_t>(particle_feedback::Destination::FilterInput))
                {
                    ++countFeedback;
                }
                else
                {
                    ++countSink;
                }
                ok = ok && !result.error;
            }

            particle_feedback::AppendRequest requests[2u]{};
            requests[0u] = particle_feedback::AppendRequest{particle_feedback::streamFilterInput(), 0u};
            requests[1u] = particle_feedback::AppendRequest{particle_feedback::streamSink(), 0u};
            for(std::uint32_t src = 0u; src < width; ++src)
            {
                requests[0u].itemCount += warp.broadcast(countFeedback, src);
                requests[1u].itemCount += warp.broadcast(countSink, src);
            }

            particle_feedback::StreamReservation<Particle> reservation{};
            auto const status = reservation.tryReserve(
                acc,
                warp,
                warp.broadcast(static_cast<std::uint32_t>(count) + 1u),
                assembler,
                requests,
                2u,
                nullptr,
                0u,
                nullptr,
                0u);
            ok = ok && status == AppendStatus::Ok;
            if(ok)
            {
                for(std::size_t i = lane; i < count; ++i)
                {
                    auto const particle = input.read(i);
                    auto const result = apply(seed, particle);
                    auto const toFeedback = static_cast<std::uint32_t>(result.destination)
                                            == static_cast<std::uint32_t>(particle_feedback::Destination::FilterInput);
                    auto const destination
                        = toFeedback ? particle_feedback::streamFilterInput() : particle_feedback::streamSink();
                    auto const group = assembler.destinationGroup(warp, destination, true);
                    auto const target = reservation.writePointer(toFeedback ? 0u : 1u, group.rank);
                    if(target == nullptr)
                    {
                        ok = false;
                        break;
                    }
                    *target = toFeedback ? feedbackParticle(particle) : particle;
                }
                ok = ok && reservation.commit(acc, warp);
                if(ok)
                    reservation.retireInput(warp);
                else
                    reservation.cancel(acc, warp);
            }
            if(!ok && error != nullptr && warp.isLeader())
                *error = true;
        }
    };

    /** Sink node: writes the final weight to the particle's unique result slot
     *  and retires the packet.
     *
     * The sink consumes its terminal batch directly (no assembly stream), so
     * `operator()` writes the output slot view in place and retires the input
     * packet exactly once, through the scheduler's own record. The per-call
     * retirement record is `*error`-adjacent bookkeeping of the caller; the
     * funnel is: one slot write per input element, indexed by the id offset in
     * the source request.
     */
    struct DestroyParticles
    {
        /** Per-particle core. Pure: the final weight is the particle weight. */
        ALPAKA_FN_HOST_ACC static constexpr particle_feedback::ApplyResult apply(Particle const& particle) noexcept
        {
            (void) particle;
            return particle_feedback::ApplyResult{particle_feedback::Destination::Sink, false};
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            JobDescriptor const& job,
            PacketAccessor const& accessor,
            StreamAssembler<Particle> const& assembler,
            bool* error) const noexcept
        {
            (void) acc;
            (void) assembler;
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();

            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            auto const count = input.size();
            auto const slots = accessor.outputSpan<std::int64_t>(job.routing.outputPort);
            auto const slotCount = slots.size();

            bool ok = true;
            for(std::size_t i = lane; i < count; ++i)
            {
                auto const particle = input.read(i);
                auto const slotIndex = particle.id - job.request.firstId;
                if(slotIndex < slotCount)
                {
                    slots.write(slotIndex, static_cast<std::int64_t>(particle.weight));
                }
                else
                {
                    ok = false;
                }
            }
            // The sink has no output stream, so there is nothing to commit. The
            // input packet is retired exactly once by the caller's scheduler
            // record, as guaranteed by the one-write-per-slot invariant above.
            (void) warp.isLeader();
            if(!ok && error != nullptr && warp.isLeader())
                *error = true;
        }
    };
} // namespace graph
