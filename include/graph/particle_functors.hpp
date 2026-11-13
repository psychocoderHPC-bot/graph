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

#include <algorithm>
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
 *   span, calls `apply()` per logical element, writes into its output spans,
 *   and reports one warp-uniform element count per port through
 *   `accessor.setOutputCount()`. The scheduler owns the destination stream:
 *   it reserves the logical range, copies the reported elements, rotates the
 *   A/B chunks, publishes full chunks, and retires the input exactly once.
 *
 *   The reserved accessor port carries the shared `ResultBlock` (per-id final
 *   weights, duplicate-sink detector, and error word). It is the same
 *   non-stream channel the toy fixture uses, so no register is hidden in a
 *   global and the same functor bodies compile on the CPU and CUDA backends.
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

        /** Error bits published into `ResultBlock::errorWord`. */
        inline constexpr std::uint32_t errorNone = 0u;
        inline constexpr std::uint32_t errorIdOutOfRange = 1u << 0u;
        inline constexpr std::uint32_t errorDuplicateSink = 1u << 1u;
        inline constexpr std::uint32_t errorWeightOverflow = 1u << 2u;
        inline constexpr std::uint32_t errorOutputOverflow = 1u << 3u;

        /** Shared result sink and error channel of the particle run.
         *
         * The pool passes this block through the reserved accessor port (the
         * same non-stream channel the toy fixture uses). Only `DestroyParticles`
         * writes the per-id weight slots; every functor may publish an error
         * bit. Slots are indexed by the source id, so the host compares one
         * final weight per generated particle.
         */
        struct ResultBlock
        {
            std::int64_t* weights{nullptr};
            std::uint32_t* sinkWrites{nullptr};
            std::uint32_t slotCount{0u};
            std::uint32_t sinkWriteCount{0u};
            std::uint32_t errorWord{0u};
            std::uint32_t reserved{0u};
        };

        static_assert(std::is_trivially_copyable_v<ResultBlock>);
        static_assert(std::is_standard_layout_v<ResultBlock>);

        ALPAKA_FN_HOST_ACC constexpr ResultBlock* resultBlock(PacketAccessor const& accessor) noexcept
        {
            return static_cast<ResultBlock*>(accessor.inputData[maxPacketPorts - 1u]);
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC inline void publishError(
            T_Acc const& acc,
            ResultBlock* control,
            std::uint32_t flags,
            bool leader) noexcept
        {
            if(control == nullptr || flags == 0u || !leader)
                return;
            (void) alpaka::onAcc::atomicOr(acc, &control->errorWord, flags, alpaka::onAcc::scope::device);
        }

        /** Checked weight increment: reports overflow instead of wrapping. */
        ALPAKA_FN_HOST_ACC constexpr bool checkedWeightAdd(std::uint32_t& weight, std::uint32_t add) noexcept
        {
            std::uint64_t const sum = static_cast<std::uint64_t>(weight) + add;
            if(sum > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
                return false;
            weight = static_cast<std::uint32_t>(sum);
            return true;
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
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
            const noexcept
        {
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const output = accessor.outputSpan<Particle>(job.routing.outputPort);
            std::uint32_t const count = job.request.count < output.size()
                                            ? static_cast<std::uint32_t>(job.request.count)
                                            : static_cast<std::uint32_t>(output.size());
            for(std::uint32_t i = lane; i < count; i += width)
                output.write(i, apply(job.request.seed, job.request.firstId + i));
            (void) accessor.setOutputCount(job.routing.outputPort, count);
        }

        /** Static `OperationFn` entry point: the pool registers a plain
         *  function pointer, not a pointer to a member template. */
        template<typename T_Acc>
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            GenerateParticles{}(acc, job, accessor);
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
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
            const noexcept
        {
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const seed = job.request.seed;
            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            std::uint32_t const count = static_cast<std::uint32_t>(input.size());
            std::uint32_t const bin0Port = job.routing.outputPort;
            std::uint32_t const bin1Port
                = job.routing.outputPortCount >= 2u ? job.routing.outputPort + 1u : job.routing.outputPort;
            auto const bin0 = accessor.outputSpan<Particle>(bin0Port);
            auto const bin1 = accessor.outputSpan<Particle>(bin1Port);
            auto* control = particle_feedback::resultBlock(accessor);

            std::uint32_t base0 = 0u;
            std::uint32_t base1 = 0u;
            std::uint32_t errors = particle_feedback::errorNone;
            for(std::uint32_t base = 0u; base < count; base += width)
            {
                std::uint32_t const index = base + lane;
                bool const active = index < count;
                Particle const particle = active ? input.read(index) : Particle{};
                auto const result = active ? apply(seed, particle) : particle_feedback::ApplyResult{};
                bool const toBin1 = active && result.destination == particle_feedback::Destination::Bin1;
                auto const mask0 = warp.ballot(active && !toBin1);
                auto const mask1 = warp.ballot(toBin1);
                std::uint32_t const rank0 = base0 + warp.rank(mask0);
                std::uint32_t const rank1 = base1 + warp.rank(mask1);
                if(active && toBin1)
                {
                    if(rank1 >= bin1.size())
                        errors |= particle_feedback::errorOutputOverflow;
                    else
                        bin1.write(rank1, particle);
                }
                else if(active)
                {
                    if(rank0 >= bin0.size())
                        errors |= particle_feedback::errorOutputOverflow;
                    else
                        bin0.write(rank0, particle);
                }
                if(active && result.error)
                    errors |= particle_feedback::errorWeightOverflow;
                base0 += WarpGroup<T_Acc>::popcount(mask0);
                base1 += WarpGroup<T_Acc>::popcount(mask1);
            }
            (void) accessor.setOutputCount(bin0Port, base0);
            if(bin1Port != bin0Port)
                (void) accessor.setOutputCount(bin1Port, base1);
            particle_feedback::publishError(acc, control, errors, warp.isLeader());
        }

        /** Static `OperationFn` entry point: the pool registers a plain
         *  function pointer, not a pointer to a member template. */
        template<typename T_Acc>
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            FilterByProperty{}(acc, job, accessor);
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
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
            const noexcept
        {
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const seed = job.request.seed;
            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            std::uint32_t const count = static_cast<std::uint32_t>(input.size());
            std::uint32_t const feedbackPort = job.routing.outputPort;
            std::uint32_t const sinkPort
                = job.routing.outputPortCount >= 2u ? job.routing.outputPort + 1u : job.routing.outputPort;
            auto const feedback = accessor.outputSpan<Particle>(feedbackPort);
            auto const sink = accessor.outputSpan<Particle>(sinkPort);
            auto* control = particle_feedback::resultBlock(accessor);

            std::uint32_t baseFeedback = 0u;
            std::uint32_t baseSink = 0u;
            std::uint32_t errors = particle_feedback::errorNone;
            for(std::uint32_t base = 0u; base < count; base += width)
            {
                std::uint32_t const index = base + lane;
                bool const active = index < count;
                Particle const particle = active ? input.read(index) : Particle{};
                auto const result = active ? apply(seed, particle) : particle_feedback::ApplyResult{};
                bool const toFeedback = active && result.destination == particle_feedback::Destination::FilterInput;
                auto const feedbackMask = warp.ballot(toFeedback);
                auto const sinkMask = warp.ballot(active && !toFeedback);
                std::uint32_t const feedbackRank = baseFeedback + warp.rank(feedbackMask);
                std::uint32_t const sinkRank = baseSink + warp.rank(sinkMask);
                if(toFeedback)
                {
                    if(feedbackRank >= feedback.size())
                        errors |= particle_feedback::errorOutputOverflow;
                    else
                        feedback.write(feedbackRank, feedbackParticle(particle));
                }
                else if(active)
                {
                    if(sinkRank >= sink.size())
                        errors |= particle_feedback::errorOutputOverflow;
                    else
                        sink.write(sinkRank, particle);
                }
                if(active && result.error)
                    errors |= particle_feedback::errorWeightOverflow;
                baseFeedback += WarpGroup<T_Acc>::popcount(feedbackMask);
                baseSink += WarpGroup<T_Acc>::popcount(sinkMask);
            }
            (void) accessor.setOutputCount(feedbackPort, baseFeedback);
            if(sinkPort != feedbackPort)
                (void) accessor.setOutputCount(sinkPort, baseSink);
            particle_feedback::publishError(acc, control, errors, warp.isLeader());
        }

        /** Static `OperationFn` entry point: the pool registers a plain
         *  function pointer, not a pointer to a member template. */
        template<typename T_Acc>
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            ProcessA{}(acc, job, accessor);
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
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
            const noexcept
        {
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const seed = job.request.seed;
            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            std::uint32_t const count = static_cast<std::uint32_t>(input.size());
            std::uint32_t const feedbackPort = job.routing.outputPort;
            std::uint32_t const sinkPort
                = job.routing.outputPortCount >= 2u ? job.routing.outputPort + 1u : job.routing.outputPort;
            auto const feedback = accessor.outputSpan<Particle>(feedbackPort);
            auto const sink = accessor.outputSpan<Particle>(sinkPort);
            auto* control = particle_feedback::resultBlock(accessor);

            std::uint32_t baseFeedback = 0u;
            std::uint32_t baseSink = 0u;
            std::uint32_t errors = particle_feedback::errorNone;
            for(std::uint32_t base = 0u; base < count; base += width)
            {
                std::uint32_t const index = base + lane;
                bool const active = index < count;
                Particle const particle = active ? input.read(index) : Particle{};
                auto const result = active ? apply(seed, particle) : particle_feedback::ApplyResult{};
                bool const toFeedback = active && result.destination == particle_feedback::Destination::FilterInput;
                auto const feedbackMask = warp.ballot(toFeedback);
                auto const sinkMask = warp.ballot(active && !toFeedback);
                std::uint32_t const feedbackRank = baseFeedback + warp.rank(feedbackMask);
                std::uint32_t const sinkRank = baseSink + warp.rank(sinkMask);
                if(toFeedback)
                {
                    if(feedbackRank >= feedback.size())
                        errors |= particle_feedback::errorOutputOverflow;
                    else
                        feedback.write(feedbackRank, feedbackParticle(particle));
                }
                else if(active)
                {
                    if(sinkRank >= sink.size())
                        errors |= particle_feedback::errorOutputOverflow;
                    else
                        sink.write(sinkRank, particle);
                }
                if(active && result.error)
                    errors |= particle_feedback::errorWeightOverflow;
                baseFeedback += WarpGroup<T_Acc>::popcount(feedbackMask);
                baseSink += WarpGroup<T_Acc>::popcount(sinkMask);
            }
            (void) accessor.setOutputCount(feedbackPort, baseFeedback);
            if(sinkPort != feedbackPort)
                (void) accessor.setOutputCount(sinkPort, baseSink);
            particle_feedback::publishError(acc, control, errors, warp.isLeader());
        }

        /** Static `OperationFn` entry point: the pool registers a plain
         *  function pointer, not a pointer to a member template. */
        template<typename T_Acc>
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            ProcessB{}(acc, job, accessor);
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
        ALPAKA_FN_ACC void operator()(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
            const noexcept
        {
            WarpGroup<T_Acc> warp{acc};
            auto const lane = warp.laneIndex();
            auto const width = warp.workerWidth();
            auto const input = accessor.inputSpan<Particle const>(job.routing.inputPort);
            std::uint32_t const count = static_cast<std::uint32_t>(input.size());
            auto* control = particle_feedback::resultBlock(accessor);
            std::uint32_t const slots
                = control != nullptr ? std::min(control->slotCount, control->sinkWriteCount) : 0u;
            std::uint32_t errors = particle_feedback::errorNone;
            for(std::uint32_t i = lane; i < count; i += width)
            {
                Particle const particle = input.read(i);
                if(control == nullptr || control->weights == nullptr || control->sinkWrites == nullptr
                   || particle.id >= static_cast<std::uint64_t>(slots))
                {
                    errors |= particle_feedback::errorIdOutOfRange;
                    continue;
                }
                auto const slot = static_cast<std::uint32_t>(particle.id);
                std::uint32_t const previous = alpaka::onAcc::atomicAdd(
                    acc,
                    &control->sinkWrites[slot],
                    std::uint32_t{1},
                    alpaka::onAcc::scope::device);
                if(previous != 0u)
                    errors |= particle_feedback::errorDuplicateSink;
                control->weights[slot] = static_cast<std::int64_t>(particle.weight);
            }
            particle_feedback::publishError(acc, control, errors, warp.isLeader());
        }

        /** Static `OperationFn` entry point: the pool registers a plain
         *  function pointer, not a pointer to a member template. */
        template<typename T_Acc>
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            DestroyParticles{}(acc, job, accessor);
        }
    };
} // namespace graph
