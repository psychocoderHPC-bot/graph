// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/detail/operation.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/views.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/graph_model.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Deterministic scripted operations for the pool-core tests.
 *
 * This is a test fixture, not part of the library. The operations carry no
 * random state, use integer arithmetic only, allocate nothing, and call no host
 * API from device code. They exist so the worker-pool, chunk-assembly, and
 * quiescence tests can drive scheduling, chunk rotation, and drain with exact
 * expected results.
 *
 * # Operation set
 *
 * | Operation        | Input                       | Output                              |
 * | ---------------- | --------------------------- | ----------------------------------- |
 * | `ToySource`      | `job.request`               | one `ToyItem` per source id          |
 * | `ToyPassThrough` | one `ToyItem` packet        | one packet, port from `routing`      |
 * | `ToySplit`       | one `ToyItem` packet        | bin 0 and bin 1, `id % 2`            |
 * | `ToyRevisit`     | one `ToyItem` packet        | feedback with `visit+1` or sink      |
 * | `ToySink`        | one `ToyItem` packet        | per-id result slot, no packet        |
 * | `ToyAbort`       | none                        | sets the diagnostic abort state      |
 *
 * Every operation is a struct with an `ALPAKA_FN_ACC` static `operation`
 * function of the project ABI:
 *
 *     void (*)(T_Acc const&, JobDescriptor const&, PacketAccessor const&)
 *
 * The ABI function is a thin wrapper. Each operation also exposes a pure,
 * host-callable core (`toySourcePlan`, `toyPassThroughCopy`, `toySplitCopy`,
 * `toyRevisitPlan`, `toySinkStore`, `toyRequestAbort`) that the unit tests
 * exercise without an accelerator. The block of the ABI wrappers is warp
 * uniform: loops and collectives run for every lane.
 *
 * # Instrumentation channel
 *
 * The ABI has no dedicated state argument. This fixture reserves port index
 * `toyControlPort` (= 3) of the packet accessor as a non-stream control channel:
 * the scheduler calls `accessor.setInput(toyControlPort, nullptr, control)`
 * with a device-visible `ToyInstrumentation`. `inputData[toyControlPort]` then
 * carries the pointer. A null control block disables instrumentation; the
 * operations still run. The channel never becomes a graph stream and does not
 * change the routing of the data ports.
 *
 * Counters are 64-bit and updated by the warp leader with a device-scope
 * atomic add, so concurrent workers never lose an update and the scheduling
 * path sees no extra dependencies. The error word is updated with a
 * device-scope atomic OR of a per-lane accumulated mask.
 */
namespace graph::toy
{
    /** One element carried by every toy stream. */
    struct ToyItem
    {
        std::uint64_t id{0u};
        std::uint32_t value{0u};
        std::uint32_t visit{0u};
    };

    static_assert(sizeof(ToyItem) == 16u, "ToyItem must stay a 16-byte POD");
    static_assert(alignof(ToyItem) == 8u);
    static_assert(std::is_trivially_copyable_v<ToyItem>);
    static_assert(std::is_standard_layout_v<ToyItem>);
    static_assert(PacketElement<ToyItem>);

    /** Per-id result slot written by `ToySink`.
     *
     * `sinkVisits` is the number of sink visits observed for this id. A test
     * reads it together with `ToyInstrumentation::sinkWrites`: value zero means
     * the id never reached the sink, value greater than one means a duplicate.
     */
    struct ToyResult
    {
        std::uint32_t value{0u};
        std::uint32_t visits{0u};
        std::uint32_t sinkVisits{0u};
        std::uint32_t reserved{0u};
    };

    static_assert(std::is_trivially_copyable_v<ToyResult>);
    static_assert(std::is_standard_layout_v<ToyResult>);

    /** Reserved accessor port of the non-stream instrumentation channel. */
    inline constexpr std::uint32_t toyControlPort = maxPacketPorts - 1u;

    /** Counter slots of the shared instrumentation block. */
    enum class ToyOpSlot : std::uint32_t
    {
        Source = 0u,
        PassThrough,
        Split,
        Revisit,
        Sink,
        Abort
    };

    inline constexpr std::uint32_t toyOpSlotCount = 6u;

    /** Error bits reported through `ToyInstrumentation::errorWord`. */
    inline constexpr std::uint32_t toyErrorNone = 0u;
    inline constexpr std::uint32_t toyErrorIdOutOfRange = 1u << 0u;
    inline constexpr std::uint32_t toyErrorRequestOverflow = 1u << 1u;
    inline constexpr std::uint32_t toyErrorVisitOverflow = 1u << 2u;
    inline constexpr std::uint32_t toyErrorDuplicateSink = 1u << 3u;
    inline constexpr std::uint32_t toyErrorOutputOverflow = 1u << 4u;
    inline constexpr std::uint32_t toyErrorAborted = 1u << 5u;

    /** Device-visible instrumentation shared by all toy operations.
     *
     * `idLimit == 0` disables the range check. Otherwise every operation flags
     * `toyErrorIdOutOfRange` for an element with `id >= idLimit`.
     * `results`, `resultCount`, `sinkWrites`, and `sinkWriteCount` describe the
     * sink storage and the per-id duplicate detector. `continuation` and
     * `continuationValid` publish the pending resumable source range.
     */
    struct ToyInstrumentation
    {
        std::uint64_t invocations[toyOpSlotCount]{};
        std::uint64_t elements[toyOpSlotCount]{};
        std::uint32_t errorWord{0u};
        std::uint32_t abortRequested{0u};
        std::uint32_t idLimit{0u};
        std::uint32_t lastEmittedCount{0u};
        RequestSpec continuation{};
        std::uint32_t continuationValid{0u};
        std::uint32_t reserved{0u};
        ToyResult* results{nullptr};
        std::uint32_t resultCount{0u};
        std::uint32_t* sinkWrites{nullptr};
        std::uint32_t sinkWriteCount{0u};
    };

    static_assert(std::is_trivially_copyable_v<ToyInstrumentation>);
    static_assert(std::is_standard_layout_v<ToyInstrumentation>);

    /** Control block carried by the reserved accessor port, or nullptr. */
    ALPAKA_FN_HOST_ACC constexpr ToyInstrumentation* toyControl(PacketAccessor const& accessor) noexcept
    {
        return static_cast<ToyInstrumentation*>(accessor.inputData[toyControlPort]);
    }

    /** Merge error bits into the instrumentation word (host-side helper). */
    ALPAKA_FN_HOST_ACC constexpr void toyMergeErrorWord(ToyInstrumentation& control, std::uint32_t flags) noexcept
    {
        control.errorWord |= flags;
    }

    /** Deterministic value of one item.
     *
     * A 64-bit integer mix of the id and the seed. There is no generator state,
     * so the value is independent of scheduling order, chunk size, and backend.
     */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t toyValue(std::uint64_t id, std::uint64_t seed) noexcept
    {
        std::uint64_t hash = id + 0x9e37'79b9'7f4a'7c15ull * (seed + 1u);
        hash ^= hash >> 30u;
        hash *= 0xbf58'476d'1ce4'e5b9ull;
        hash ^= hash >> 27u;
        hash *= 0x94d0'49bb'1331'11ebull;
        hash ^= hash >> 31u;
        return static_cast<std::uint32_t>(hash ^ (hash >> 32u));
    }

    /** Build the item of one source id. */
    ALPAKA_FN_HOST_ACC constexpr ToyItem toyMakeItem(
        std::uint64_t id,
        std::uint64_t seed,
        std::uint32_t visit = 0u) noexcept
    {
        return ToyItem{id, toyValue(id, seed), visit};
    }

    /** Id range check used by every operation. */
    ALPAKA_FN_HOST_ACC constexpr bool toyIdInRange(std::uint64_t id, std::uint32_t idLimit) noexcept
    {
        return idLimit == 0u || id < static_cast<std::uint64_t>(idLimit);
    }

    ALPAKA_FN_HOST_ACC constexpr std::uint32_t toyMin(std::uint32_t lhs, std::uint32_t rhs) noexcept
    {
        return lhs < rhs ? lhs : rhs;
    }

    /** Result of planning one `ToySource` invocation. */
    struct ToySourcePlan
    {
        /** Number of items this invocation may emit. */
        std::uint32_t emitted{0u};
        /** Error bits collected while planning. */
        std::uint32_t errorFlags{toyErrorNone};
        /** Resumable range of the remaining items; `count == 0` means done. */
        RequestSpec continuation{};
    };

    /** Pure source planning core.
     *
     * Emits `min(count, capacity, maxPerInvocation)` items and returns the
     * remaining id range as a `RequestSpec`. A continuation with `count == 0`
     * is complete. The request is emitted unchanged in that case. A request
     * whose `firstId + emitted` would wrap reports `toyErrorRequestOverflow`
     * and makes no progress.
     *
     * Contract: `capacity` is the number of `ToyItem` elements the caller
     * allocated for the output buffer. The core never plans more items than
     * that declared capacity. A request that needs more items than `capacity`
     * for one invocation is clamped to `capacity` and reports
     * `toyErrorOutputOverflow`; the unemitted remainder stays in the
     * continuation.
     */
    ALPAKA_FN_HOST_ACC constexpr ToySourcePlan toySourcePlan(
        RequestSpec const& request,
        std::uint32_t capacity,
        std::uint32_t maxPerInvocation) noexcept
    {
        ToySourcePlan plan{};
        plan.continuation = request;
        if(request.count == 0u)
            return plan;

        // Number of items this invocation would emit before the
        // caller-declared output capacity limits it.
        std::uint64_t const requestable = request.count < static_cast<std::uint64_t>(maxPerInvocation)
                                              ? request.count
                                              : static_cast<std::uint64_t>(maxPerInvocation);
        if(requestable > static_cast<std::uint64_t>(capacity))
            plan.errorFlags |= toyErrorOutputOverflow;

        std::uint32_t const limit = toyMin(capacity, maxPerInvocation);
        if(limit == 0u)
            return plan;

        std::uint64_t const wanted
            = request.count < static_cast<std::uint64_t>(limit) ? request.count : static_cast<std::uint64_t>(limit);
        auto const emitted = static_cast<std::uint32_t>(wanted);
        if(request.firstId > static_cast<std::uint64_t>(~std::uint64_t{0u}) - emitted)
        {
            plan.errorFlags |= toyErrorRequestOverflow;
            plan.continuation.count = 0u;
            return plan;
        }

        plan.emitted = emitted;
        plan.continuation.firstId = request.firstId + emitted;
        plan.continuation.count = request.count - emitted;
        plan.continuation.seed = request.seed;
        return plan;
    }

    /** Pure source core: plan one invocation and write its items.
     *
     * Contract: `capacity` is the number of `ToyItem` elements the caller
     * allocated at `out`. This function writes at most that many elements and
     * never past the declared bound. An undersized capacity is reported as
     * `toyErrorOutputOverflow` by the planner, not by an out-of-bounds write.
     */
    ALPAKA_FN_HOST_ACC constexpr ToySourcePlan toySourceEmit(
        RequestSpec const& request,
        ToyItem* out,
        std::uint32_t capacity,
        std::uint32_t maxPerInvocation,
        std::uint32_t idLimit) noexcept
    {
        ToySourcePlan plan = toySourcePlan(request, capacity, maxPerInvocation);
        if(out == nullptr)
            return plan;
        std::uint32_t const writeCount = toyMin(plan.emitted, capacity);
        for(std::uint32_t i = 0u; i < writeCount; ++i)
        {
            std::uint64_t const id = request.firstId + i;
            if(!toyIdInRange(id, idLimit))
                plan.errorFlags |= toyErrorIdOutOfRange;
            out[i] = toyMakeItem(id, request.seed);
        }
        return plan;
    }

    /** Output port selected by a pass-through job.
     *
     * This is the explicit routing field: a job targets either bin by setting
     * `RoutingInfo::outputPort`.
     */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t toyPassThroughPort(RoutingInfo const& routing) noexcept
    {
        return routing.outputPort;
    }

    /** Result of a pass-through or split copy. */
    struct ToyCopyOutcome
    {
        std::uint32_t copied{0u};
        std::uint32_t errorFlags{toyErrorNone};
    };

    /** Pure pass-through core: copy `count` items unchanged, at most `capacity`. */
    ALPAKA_FN_HOST_ACC constexpr ToyCopyOutcome toyPassThroughCopy(
        ToyItem const* in,
        std::uint32_t count,
        ToyItem* out,
        std::uint32_t capacity,
        std::uint32_t idLimit) noexcept
    {
        ToyCopyOutcome outcome{};
        std::uint32_t const copied = toyMin(count, capacity);
        if(count > capacity)
            outcome.errorFlags |= toyErrorOutputOverflow;
        if(in == nullptr || out == nullptr)
            return outcome;
        for(std::uint32_t i = 0u; i < copied; ++i)
        {
            if(!toyIdInRange(in[i].id, idLimit))
                outcome.errorFlags |= toyErrorIdOutOfRange;
            out[i] = in[i];
        }
        outcome.copied = copied;
        return outcome;
    }

    /** Destination of one item: bin 0 for an even id, bin 1 for an odd id. */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t toySplitBin(ToyItem const& item) noexcept
    {
        return static_cast<std::uint32_t>(item.id & 1u);
    }

    /** Rank of `index` inside its destination group of `items[0..count)`.
     *
     * Counts the earlier items that target the same bin. This is the scalar
     * equivalent of the ballot rank used by the ABI wrapper.
     */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t toySplitRank(
        ToyItem const* items,
        std::uint32_t count,
        std::uint32_t index) noexcept
    {
        if(items == nullptr || index >= count)
            return 0u;
        std::uint32_t const bin = toySplitBin(items[index]);
        std::uint32_t rank = 0u;
        for(std::uint32_t i = 0u; i < index; ++i)
        {
            if(toySplitBin(items[i]) == bin)
                ++rank;
        }
        return rank;
    }

    /** Result of a split copy. */
    struct ToySplitOutcome
    {
        std::uint32_t bin0Count{0u};
        std::uint32_t bin1Count{0u};
        std::uint32_t errorFlags{toyErrorNone};
    };

    /** Pure split core: compact both destination groups independently. */
    ALPAKA_FN_HOST_ACC constexpr ToySplitOutcome toySplitCopy(
        ToyItem const* in,
        std::uint32_t count,
        ToyItem* bin0,
        std::uint32_t bin0Capacity,
        ToyItem* bin1,
        std::uint32_t bin1Capacity,
        std::uint32_t idLimit) noexcept
    {
        ToySplitOutcome outcome{};
        if(in == nullptr)
            return outcome;
        for(std::uint32_t i = 0u; i < count; ++i)
        {
            ToyItem const item = in[i];
            if(!toyIdInRange(item.id, idLimit))
                outcome.errorFlags |= toyErrorIdOutOfRange;
            if(toySplitBin(item) == 0u)
            {
                if(outcome.bin0Count < bin0Capacity && bin0 != nullptr)
                    bin0[outcome.bin0Count] = item;
                else
                    outcome.errorFlags |= toyErrorOutputOverflow;
                ++outcome.bin0Count;
            }
            else
            {
                if(outcome.bin1Count < bin1Capacity && bin1 != nullptr)
                    bin1[outcome.bin1Count] = item;
                else
                    outcome.errorFlags |= toyErrorOutputOverflow;
                ++outcome.bin1Count;
            }
        }
        return outcome;
    }

    /** Decision of one revisit pass. */
    struct ToyRevisitOutcome
    {
        /** True: the item returns to feedback. False: the item goes to the sink. */
        bool toFeedback{false};
        /** Item after applying the revisit decision. */
        ToyItem item{};
        std::uint32_t errorFlags{toyErrorNone};
    };

    /** Pure revisit core.
     *
     * While `visit < revisitCount` the item returns to feedback with
     * `visit + 1`. Otherwise the item goes to the sink unchanged. An item that
     * starts at visit 0 therefore returns exactly `revisitCount` times and
     * reaches the sink with `visit == revisitCount`.
     */
    ALPAKA_FN_HOST_ACC constexpr ToyRevisitOutcome toyRevisitPlan(
        ToyItem item,
        std::uint32_t revisitCount,
        std::uint32_t idLimit) noexcept
    {
        ToyRevisitOutcome outcome{};
        outcome.item = item;
        if(!toyIdInRange(item.id, idLimit))
            outcome.errorFlags |= toyErrorIdOutOfRange;
        if(item.visit < revisitCount)
        {
            if(item.visit == ~std::uint32_t{0u})
            {
                outcome.errorFlags |= toyErrorVisitOverflow;
                outcome.toFeedback = false;
                return outcome;
            }
            outcome.toFeedback = true;
            outcome.item.visit = item.visit + 1u;
        }
        return outcome;
    }

    /** Result of one sink store. */
    struct ToySinkOutcome
    {
        std::uint32_t errorFlags{toyErrorNone};
        /** Counter value before this store; nonzero means a duplicate visit. */
        std::uint32_t previousWrites{0u};
        bool stored{false};
    };

    /** Pure sink core: store one result and count the visit for its id.
     *
     * The per-id counter is incremented for every store. A nonzero previous
     * value sets `toyErrorDuplicateSink`. An id outside `resultCount` or
     * `writeCount` sets `toyErrorIdOutOfRange` and writes nothing.
     */
    ALPAKA_FN_HOST_ACC constexpr ToySinkOutcome toySinkStore(
        ToyItem const& item,
        ToyResult* results,
        std::uint32_t resultCount,
        std::uint32_t* writes,
        std::uint32_t writeCount) noexcept
    {
        ToySinkOutcome outcome{};
        std::uint64_t const id = item.id;
        if(results == nullptr || writes == nullptr || id >= static_cast<std::uint64_t>(resultCount)
           || id >= static_cast<std::uint64_t>(writeCount))
        {
            outcome.errorFlags |= toyErrorIdOutOfRange;
            return outcome;
        }
        auto const slot = static_cast<std::uint32_t>(id);
        outcome.previousWrites = writes[slot];
        writes[slot] = writes[slot] + 1u;
        if(outcome.previousWrites != 0u)
            outcome.errorFlags |= toyErrorDuplicateSink;
        results[slot] = ToyResult{item.value, item.visit, outcome.previousWrites + 1u, 0u};
        outcome.stored = true;
        return outcome;
    }

    /** Pure abort core: request a diagnostic abort and mark the error word.
     *
     * The abort state is distinct from a successful drain. The drain path never
     * writes it.
     */
    ALPAKA_FN_HOST_ACC constexpr void toyRequestAbort(ToyInstrumentation& control) noexcept
    {
        control.abortRequested = 1u;
        control.errorWord |= toyErrorAborted;
    }

    /** Record one invocation and its element count on the warp leader.
     *
     * Device-scope atomic add: safe for concurrent workers, and the counters
     * are not part of any scheduling condition.
     */
    template<typename T_Acc>
    ALPAKA_FN_ACC void toyRecordInvocation(
        T_Acc const& acc,
        ToyInstrumentation* control,
        ToyOpSlot slot,
        std::uint32_t elementCount) noexcept
    {
        if(control == nullptr)
            return;
        WarpGroup<T_Acc> const warp{acc};
        if(warp.isLeader())
        {
            auto const index = static_cast<std::uint32_t>(slot);
            if(elementCount != 0u)
            {
                (void) alpaka::onAcc::atomicAdd(
                    acc,
                    &control->elements[index],
                    static_cast<std::uint64_t>(elementCount),
                    alpaka::onAcc::scope::device);
            }
            (void) alpaka::onAcc::atomicAdd(
                acc,
                &control->invocations[index],
                std::uint64_t{1},
                alpaka::onAcc::scope::device);
        }
    }

    /** OR a per-lane error mask into the shared error word. */
    template<typename T_Acc>
    ALPAKA_FN_ACC void toyPublishErrors(
        T_Acc const& acc,
        ToyInstrumentation* control,
        std::uint32_t errorFlags) noexcept
    {
        if(control == nullptr || errorFlags == toyErrorNone)
            return;
        (void) alpaka::onAcc::atomicOr(acc, &control->errorWord, errorFlags, alpaka::onAcc::scope::device);
    }

    /** Source operation.
     *
     * Reads `job.request`, emits at most one output chunk per invocation, and
     * publishes the resumable remainder in `control->continuation`. The emitted
     * count is published in `control->lastEmittedCount` so the test scheduler
     * can size the output packet.
     */
    template<typename T_Acc>
    struct ToySource
    {
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            auto* control = toyControl(accessor);
            auto const output = accessor.outputSpan<ToyItem>(job.routing.outputPort);
            auto const* record = accessor.outputRecord(job.routing.outputPort);
            std::uint32_t const capacity
                = record != nullptr ? record->capacity : static_cast<std::uint32_t>(output.size());
            WarpGroup<T_Acc> const warp{acc};

            ToySourcePlan plan{};
            if(warp.isLeader())
            {
                std::uint32_t const idLimit = control != nullptr ? control->idLimit : 0u;
                plan = toySourcePlan(job.request, capacity, capacity);
                if(control != nullptr)
                {
                    control->lastEmittedCount = plan.emitted;
                    control->continuation = plan.continuation;
                    control->continuationValid = plan.continuation.count != 0u ? 1u : 0u;
                }
                // The id range check is part of the pure core, so replay it here.
                for(std::uint32_t i = 0u; i < plan.emitted; ++i)
                {
                    if(!toyIdInRange(job.request.firstId + i, idLimit))
                        plan.errorFlags |= toyErrorIdOutOfRange;
                }
            }
            std::uint32_t const emitted = warp.template broadcast<std::uint32_t>(plan.emitted, 0u);
            std::uint32_t const idLimit = control != nullptr ? control->idLimit : 0u;
            std::uint32_t const width = warp.workerWidth();
            std::uint32_t localErrors = toyErrorNone;
            for(std::uint32_t i = warp.laneIndex(); i < emitted; i += width)
            {
                std::uint64_t const id = job.request.firstId + i;
                if(!toyIdInRange(id, idLimit))
                    localErrors |= toyErrorIdOutOfRange;
                output.write(i, toyMakeItem(id, job.request.seed));
            }
            toyPublishErrors(acc, control, localErrors | plan.errorFlags);
            toyRecordInvocation(acc, control, ToyOpSlot::Source, emitted);
        }
    };

    /** Pass-through operation: copy the input unchanged to the routed output. */
    template<typename T_Acc>
    struct ToyPassThrough
    {
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            auto* control = toyControl(accessor);
            auto const input = accessor.inputSpan<ToyItem>(job.routing.inputPort);
            auto const output = accessor.outputSpan<ToyItem>(toyPassThroughPort(job.routing));
            std::uint32_t const count = static_cast<std::uint32_t>(input.size());
            std::uint32_t const idLimit = control != nullptr ? control->idLimit : 0u;
            WarpGroup<T_Acc> const warp{acc};
            std::uint32_t const width = warp.workerWidth();
            std::uint32_t localErrors = toyErrorNone;
            if(count > output.size())
                localErrors |= toyErrorOutputOverflow;
            for(std::uint32_t i = warp.laneIndex(); i < count; i += width)
            {
                ToyItem item = input.read(i);
                if(!toyIdInRange(item.id, idLimit))
                    localErrors |= toyErrorIdOutOfRange;
                output.write(i, item);
            }
            toyPublishErrors(acc, control, localErrors);
            toyRecordInvocation(acc, control, ToyOpSlot::PassThrough, count);
        }
    };

    /** Split operation: route by `id % 2` to the two consecutive output ports. */
    template<typename T_Acc>
    struct ToySplit
    {
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            auto* control = toyControl(accessor);
            auto const input = accessor.inputSpan<ToyItem>(job.routing.inputPort);
            auto const bin0 = accessor.outputSpan<ToyItem>(job.routing.outputPort);
            std::uint32_t const bin1Port
                = job.routing.outputPortCount >= 2u ? job.routing.outputPort + 1u : job.routing.outputPort;
            auto const bin1 = accessor.outputSpan<ToyItem>(bin1Port);
            bool const hasBin1 = job.routing.outputPortCount >= 2u;
            std::uint32_t const count = static_cast<std::uint32_t>(input.size());
            std::uint32_t const idLimit = control != nullptr ? control->idLimit : 0u;

            WarpGroup<T_Acc> const warp{acc};
            std::uint32_t const width = warp.workerWidth();
            std::uint32_t base0 = 0u;
            std::uint32_t base1 = 0u;
            std::uint32_t localErrors = toyErrorNone;
            for(std::uint32_t base = 0u; base < count; base += width)
            {
                std::uint32_t const index = base + warp.laneIndex();
                bool const active = index < count;
                ToyItem const item = active ? input.read(index) : ToyItem{};
                std::uint32_t const bin = active ? toySplitBin(item) : 0u;
                auto const mask0 = warp.ballot(active && bin == 0u);
                auto const mask1 = warp.ballot(active && bin == 1u);
                std::uint32_t const rank = bin == 0u ? base0 + warp.rank(mask0) : base1 + warp.rank(mask1);
                if(active)
                {
                    if(!toyIdInRange(item.id, idLimit))
                        localErrors |= toyErrorIdOutOfRange;
                    if(bin == 0u)
                    {
                        if(rank >= bin0.size())
                            localErrors |= toyErrorOutputOverflow;
                        bin0.write(rank, item);
                    }
                    else
                    {
                        if(!hasBin1 || rank >= bin1.size())
                            localErrors |= toyErrorOutputOverflow;
                        bin1.write(rank, item);
                    }
                }
                base0 += WarpGroup<T_Acc>::popcount(mask0);
                base1 += WarpGroup<T_Acc>::popcount(mask1);
            }
            toyPublishErrors(acc, control, localErrors);
            toyRecordInvocation(acc, control, ToyOpSlot::Split, count);
        }
    };

    /** Revisit operation: feedback with `visit + 1` for `revisitCount` rounds. */
    template<typename T_Acc>
    struct ToyRevisit
    {
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            auto* control = toyControl(accessor);
            auto const input = accessor.inputSpan<ToyItem>(job.routing.inputPort);
            bool const hasFeedback = job.routing.outputPortCount >= 2u;
            std::uint32_t const feedbackPort = job.routing.outputPort;
            std::uint32_t const sinkPort = hasFeedback ? job.routing.outputPort + 1u : job.routing.outputPort;
            auto const feedback = accessor.outputSpan<ToyItem>(feedbackPort);
            auto const sink = accessor.outputSpan<ToyItem>(sinkPort);
            auto const revisitCount = static_cast<std::uint32_t>(job.request.count);
            std::uint32_t const count = static_cast<std::uint32_t>(input.size());
            std::uint32_t const idLimit = control != nullptr ? control->idLimit : 0u;

            WarpGroup<T_Acc> const warp{acc};
            std::uint32_t const width = warp.workerWidth();
            std::uint32_t baseFeedback = 0u;
            std::uint32_t baseSink = 0u;
            std::uint32_t localErrors = toyErrorNone;
            for(std::uint32_t base = 0u; base < count; base += width)
            {
                std::uint32_t const index = base + warp.laneIndex();
                bool const active = index < count;
                ToyItem const item = active ? input.read(index) : ToyItem{};
                ToyRevisitOutcome outcome{};
                if(active)
                    outcome = toyRevisitPlan(item, revisitCount, idLimit);
                bool const toFeedback = active && outcome.toFeedback;
                bool const toSink = active && !outcome.toFeedback;
                auto const feedbackMask = warp.ballot(toFeedback);
                auto const sinkMask = warp.ballot(toSink);
                std::uint32_t const feedbackRank = baseFeedback + warp.rank(feedbackMask);
                std::uint32_t const sinkRank = baseSink + warp.rank(sinkMask);
                if(active)
                {
                    localErrors |= outcome.errorFlags;
                    if(toFeedback)
                    {
                        if(feedbackRank >= feedback.size())
                            localErrors |= toyErrorOutputOverflow;
                        feedback.write(feedbackRank, outcome.item);
                    }
                    else
                    {
                        if(sinkRank >= sink.size())
                            localErrors |= toyErrorOutputOverflow;
                        sink.write(sinkRank, outcome.item);
                    }
                }
                baseFeedback += WarpGroup<T_Acc>::popcount(feedbackMask);
                baseSink += WarpGroup<T_Acc>::popcount(sinkMask);
            }
            toyPublishErrors(acc, control, localErrors);
            toyRecordInvocation(acc, control, ToyOpSlot::Revisit, count);
        }
    };

    /** Sink operation: store the final value and count visits per id. */
    template<typename T_Acc>
    struct ToySink
    {
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
        {
            auto* control = toyControl(accessor);
            auto const input = accessor.inputSpan<ToyItem>(job.routing.inputPort);
            std::uint32_t const count = static_cast<std::uint32_t>(input.size());
            std::uint32_t const slots
                = control != nullptr ? toyMin(control->resultCount, control->sinkWriteCount) : 0u;
            WarpGroup<T_Acc> const warp{acc};
            std::uint32_t const width = warp.workerWidth();
            std::uint32_t localErrors = toyErrorNone;
            for(std::uint32_t i = warp.laneIndex(); i < count; i += width)
            {
                ToyItem const item = input.read(i);
                if(control == nullptr || control->results == nullptr || control->sinkWrites == nullptr
                   || item.id >= static_cast<std::uint64_t>(slots))
                {
                    localErrors |= toyErrorIdOutOfRange;
                    continue;
                }
                auto const slot = static_cast<std::uint32_t>(item.id);
                auto const previous = alpaka::onAcc::atomicAdd(
                    acc,
                    &control->sinkWrites[slot],
                    std::uint32_t{1},
                    alpaka::onAcc::scope::device);
                if(previous != 0u)
                    localErrors |= toyErrorDuplicateSink;
                control->results[slot] = ToyResult{item.value, item.visit, previous + 1u, 0u};
            }
            toyPublishErrors(acc, control, localErrors);
            toyRecordInvocation(acc, control, ToyOpSlot::Sink, count);
        }
    };

    /** Control operation: request a diagnostic abort. */
    template<typename T_Acc>
    struct ToyAbort
    {
        ALPAKA_FN_ACC static void operation(T_Acc const& acc, JobDescriptor const&, PacketAccessor const& accessor)
        {
            auto* control = toyControl(accessor);
            WarpGroup<T_Acc> const warp{acc};
            if(control != nullptr && warp.isLeader())
            {
                (void) alpaka::onAcc::atomicExch(
                    acc,
                    &control->abortRequested,
                    std::uint32_t{1},
                    alpaka::onAcc::scope::device);
                std::uint32_t const abortFlags = toyErrorAborted;
                (void) alpaka::onAcc::atomicOr(acc, &control->errorWord, abortFlags, alpaka::onAcc::scope::device);
            }
            toyRecordInvocation(acc, control, ToyOpSlot::Abort, 0u);
        }
    };

    /** Operation-id registry of the toy graph. */
    struct ToyOperationIds
    {
        OperationId source{OperationId{1u}};
        OperationId split{OperationId{2u}};
        OperationId revisit{OperationId{3u}};
        OperationId sink{OperationId{4u}};
        OperationId abort{OperationId{5u}};
    };

    inline constexpr ToyOperationIds defaultToyOperationIds{};

    /** Logical stream ids of the toy graph. */
    struct ToyStreamIds
    {
        StreamId splitInput{StreamId{1u}};
        StreamId bin0{StreamId{2u}};
        StreamId bin1{StreamId{3u}};
        StreamId feedback{StreamId{4u}};
        StreamId sink{StreamId{5u}};
    };

    inline constexpr ToyStreamIds defaultToyStreamIds{};

    /** Shape of the toy graph produced by `makeToyGraph()`. */
    struct ToyGraphConfig
    {
        /** Expected number of source items; carried for the test that submits
         * the request. Topology does not depend on it. */
        std::uint64_t itemCount{256u};
        /** Number of elements per physical chunk (`Y`). */
        std::uint32_t chunkSize{128u};
        /** Number of feedback rounds applied by `ToyRevisit`. */
        std::uint32_t revisitCount{3u};
        /** Add the declared feedback edge and the revisit feedback port. */
        bool feedback{true};
        ToyOperationIds operations{};
        ToyStreamIds streams{};
    };

    inline constexpr std::uint32_t toyAssemblyChunkCount = 2u;
    inline constexpr std::uint32_t toyTerminalChunkCount = 1u;
    inline constexpr std::uint32_t toyStreamPort = 0u;

    /** Build the scripted toy graph description.
     *
     * Feedback variant (`feedback == true`):
     *
     *     ToySource -> splitInput -> ToySplit -> bin0, bin1 -> ToyRevisit
     *     ToyRevisit -> feedback -> ToySplit.input[1]   (declared feedback edge)
     *     ToyRevisit -> sink -> ToySink
     *
     * Acyclic variant (`feedback == false`): the same topology without the
     * feedback stream and edge. `ToyRevisit` then exposes only its sink port and
     * sends every item straight to the sink.
     *
     * `chunkSize == 0` returns an empty model, which `buildGraph()` rejects.
     */
    inline GraphModel makeToyGraph(ToyGraphConfig const& config = {})
    {
        GraphModel model;
        if(config.chunkSize == 0u)
            return model;

        auto const schema
            = makeElementSchema<ToyItem>(alpaka::Vec<std::size_t, 1u>{static_cast<std::size_t>(config.chunkSize)});
        std::uint32_t const port = toyStreamPort;

        model.addStream(StreamDescriptor{config.streams.splitInput, port, config.chunkSize, toyAssemblyChunkCount});
        model.addStream(StreamDescriptor{config.streams.bin0, port, config.chunkSize, toyAssemblyChunkCount});
        model.addStream(StreamDescriptor{config.streams.bin1, port, config.chunkSize, toyAssemblyChunkCount});
        if(config.feedback)
        {
            model.addStream(StreamDescriptor{config.streams.feedback, port, config.chunkSize, toyAssemblyChunkCount});
        }
        model.addStream(StreamDescriptor{config.streams.sink, port, config.chunkSize, toyTerminalChunkCount});

        NodeDescriptor source;
        source.name = "ToySource";
        source.operation = config.operations.source;
        source.outputs.push_back(OutputPortDescriptor{schema, config.streams.splitInput, port});

        NodeDescriptor split;
        split.name = "ToySplit";
        split.operation = config.operations.split;
        split.inputs.push_back(InputPortDescriptor{schema});
        if(config.feedback)
            split.inputs.push_back(InputPortDescriptor{schema});
        split.outputs.push_back(OutputPortDescriptor{schema, config.streams.bin0, port});
        split.outputs.push_back(OutputPortDescriptor{schema, config.streams.bin1, port});

        NodeDescriptor revisit;
        revisit.name = "ToyRevisit";
        revisit.operation = config.operations.revisit;
        revisit.inputs.push_back(InputPortDescriptor{schema});
        revisit.inputs.push_back(InputPortDescriptor{schema});
        revisit.outputs.push_back(OutputPortDescriptor{schema, config.streams.sink, port});
        if(config.feedback)
            revisit.outputs.push_back(OutputPortDescriptor{schema, config.streams.feedback, port});

        NodeDescriptor sink;
        sink.name = "ToySink";
        sink.operation = config.operations.sink;
        sink.inputs.push_back(InputPortDescriptor{schema});

        auto const sourceNode = model.addNode(std::move(source));
        auto const splitNode = model.addNode(std::move(split));
        auto const revisitNode = model.addNode(std::move(revisit));
        auto const sinkNode = model.addNode(std::move(sink));

        model.addEdge(EdgeDescriptor{sourceNode, 0u, splitNode, 0u, config.streams.splitInput, port, false});
        model.addEdge(EdgeDescriptor{splitNode, 0u, revisitNode, 0u, config.streams.bin0, port, false});
        model.addEdge(EdgeDescriptor{splitNode, 1u, revisitNode, 1u, config.streams.bin1, port, false});
        model.addEdge(EdgeDescriptor{revisitNode, 0u, sinkNode, 0u, config.streams.sink, port, false});
        if(config.feedback)
        {
            model.addEdge(EdgeDescriptor{revisitNode, 1u, splitNode, 1u, config.streams.feedback, port, true});
        }

        return model;
    }
} // namespace graph::toy
