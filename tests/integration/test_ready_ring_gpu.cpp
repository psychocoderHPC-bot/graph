// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/warp_barrier.hpp>
#include <graph/detail/warp_group.hpp>
#include <graph/ready_ring.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>

/** Ready-ring cross-warp visibility integration test.
 *
 * One warp per block (the bring-up layout). Blocks `[0, numBlocks/2)` are
 * producers, blocks `[numBlocks/2, numBlocks)` are consumers. Each producer
 * pushes `entriesPerBlock` items; each consumer dequeues `entriesPerBlock`
 * items. Total items pushed == total items popped.
 *
 * This is the spec's cross-warp visibility row: data written by all lanes of a
 * producing warp must be fully visible to a *different* warp after dequeue. A
 * device-scope release on the publish and a device-scope acquire on the claim
 * carry the publication chain (validated by P1: 12.4 M msg/s cross block with
 * zero word errors).
 *
 * Correctness argument (no unbounded spin, no lap):
 *   - All blocks first rendezvous on a global arrival counter so the ring is
 *     initialized before any producer pushes.
 *   - The ring capacity (64) is larger than `entriesPerBlock` for the sizes
 *     used here (<=200/2), and producers and consumers are paced 1:1, so a
 *     consumer can never lap a producer: the producer for item X pushes X
 *     before it pushes X+1, and the consumer only claims up to the number of
 *     items already pushed. The per-item retry budget is therefore generous,
 *     not a correctness mechanism.
 *
 * CUDA only: the CPU serial executor runs one logical thread and cannot
 * demonstrate cross-warp visibility.
 */
#if GRAPH_HAS_CUDA
namespace
{
    using Entry = graph::ReadyEntry;
    using Word = std::uint64_t;

    /** Mix a 64-bit value (splitmix64 finalizer). */
    ALPAKA_FN_HOST_ACC constexpr Word mix64(Word x) noexcept
    {
        x ^= x >> 33u;
        x *= 0xff51'afd7'ed55'8ccdull;
        x ^= x >> 33u;
        x *= 0xc4ce'b9fe'1a85'ec53ull;
        x ^= x >> 33u;
        return x;
    }

    /** One entry's payload: 32 words, one per lane. */
    constexpr std::size_t wordsPerEntry = 32u;

    /** Global arrival-barrier state, appended after the ring storage.
     *  `ready` is incremented by each block's leader on arrival; `go` is set
     *  by the first block to see `ready == numBlocks`, and every block spins
     *  (bounded) on `go` before doing any ring work. */
    struct Arrival
    {
        std::uint64_t ready;
        std::uint64_t padReady[7]{};
        std::uint64_t go;
        std::uint64_t padGo[7]{};
    };

    /** Build a ring entry for item `it`. */
    ALPAKA_FN_ACC constexpr Entry makeEntry(Word it) noexcept
    {
        Entry e{};
        e.operation = graph::OperationId{static_cast<graph::OperationId::ValueType>(it % 32u)};
        e.kind = graph::JobKind::OPERATION;
        e.packet.index = static_cast<std::uint32_t>(it % 16u);
        e.packet.generation = static_cast<std::uint32_t>(it % 7u);
        e.routing.inputPort = 0u;
        e.routing.outputPort = static_cast<std::uint32_t>(it % 4u);
        e.routing.outputPortCount = 1u;
        e.request.firstId = it;
        e.request.count = wordsPerEntry;
        e.request.seed = mix64(it * 131u);
        e.sequenceNumber = it;
        return e;
    }

    struct CrossWarpKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto ringStorage,
            alpaka::concepts::IMdSpan auto payloadStorage,
            alpaka::concepts::IMdSpan auto claimedStorage,
            alpaka::concepts::IMdSpan auto arrivalStorage,
            alpaka::concepts::IMdSpan auto results,
            std::uint32_t numBlocks,
            std::uint64_t entriesPerBlock) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            using Ring = graph::ReadyRing<Acc, 64u>; // ring capacity 64

            auto* base = static_cast<std::uint8_t*>(ringStorage.data());
            Ring ring{base};
            auto* payload = reinterpret_cast<Word*>(payloadStorage.data());
            auto* claimed = reinterpret_cast<std::uint64_t*>(claimedStorage.data());
            auto* arrival = reinterpret_cast<Arrival*>(arrivalStorage.data());
            graph::WarpGroup<Acc> warp{acc};
            graph::WarpBarrier<Acc> barrier{};
            constexpr std::uint32_t blockThreads = 32u;
            // 1-D indexing helper for the IMdSpan results buffer.
            auto idx = [](std::size_t i) { return alpaka::Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(i)}; };
            // The grid is one-dimensional (numBlocks blocks), so the block
            // index vector has one component that is the block id.
            auto const blockIdx = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks);
            auto const blockId = static_cast<std::uint32_t>(blockIdx[0]);
            auto const lane = warp.laneIndex();

            auto const numProducers = numBlocks / 2u;
            bool const isProducer = blockId < numProducers;

            // --- Global arrival barrier: initialize the ring, then go. ---
            // Block 0's leader initializes the ring; a device-scope release
            // fence orders those stores before its arrival store. Every block
            // leader then spins on `ready == numBlocks` (a plain volatile
            // load is a bounded poll) before the block proceeds.
            if(warp.isLeader())
            {
                if(blockId == 0u)
                    ring.initialize(acc, warp);
                alpaka::onAcc::memFence(acc, alpaka::onAcc::scope::device, alpaka::onAcc::order::release);
                alpaka::onAcc::atomicAdd(acc, &arrival->ready, std::uint64_t{1u}, alpaka::onAcc::scope::device);
            }
            barrier.barrier(acc, blockThreads, warp);
            if(warp.isLeader())
            {
                std::uint64_t guard = 0u;
                while(reinterpret_cast<std::uint64_t const volatile&>(arrival->ready) < numBlocks
                      && guard++ < 1u << 26u)
                {
                    // bounded poll
                }
            }
            barrier.barrier(acc, blockThreads, warp);

            // Unique global item id for this block's item: producerIndex
            // (blockId for a producer) times entriesPerBlock plus the local
            // item, so no two items share a sequence number.
            auto const globalBase = static_cast<std::uint64_t>(blockId) * entriesPerBlock;

            if(isProducer)
            {
                std::uint64_t pushed = 0u;
                std::uint64_t pushFails = 0u;
                for(std::uint64_t it = 0u; it < entriesPerBlock; ++it)
                {
                    auto const globalItem = globalBase + it;
                    // All lanes write their word for this item.
                    payload[globalItem * wordsPerEntry + lane] = mix64(globalItem * 131u + lane);
                    barrier.barrier(acc, blockThreads, warp);

                    std::uint64_t guard = 0u;
                    auto status = ring.enqueue(acc, warp, makeEntry(globalItem));
                    while(static_cast<std::uint32_t>(status) != 0u && guard++ < 1u << 22u)
                        status = ring.enqueue(acc, warp, makeEntry(globalItem));
                    barrier.barrier(acc, blockThreads, warp);

                    if(warp.isLeader())
                    {
                        if(static_cast<std::uint32_t>(status) == 0u)
                            pushed++;
                        else
                            pushFails++;
                    }
                }
                if(warp.isLeader())
                {
                    results[idx(numProducers + 1u)] += pushed;
                    results[idx(numProducers + 6u)] += pushFails;
                }
            }
            else
            {
                auto const consumerIndex = blockId - numProducers;
                std::uint64_t claimedCount = 0u;
                std::uint64_t wordErrors = 0u;
                std::uint64_t duplicates = 0u;
                std::uint64_t timeouts = 0u;

                for(std::uint64_t it = 0u; it < entriesPerBlock; ++it)
                {
                    graph::ReadyEntry out{};
                    std::uint64_t guard = 0u;
                    auto status = ring.dequeue(acc, warp, out);
                    while(static_cast<std::uint32_t>(status) != 0u && guard++ < 1u << 22u)
                        status = ring.dequeue(acc, warp, out);
                    barrier.barrier(acc, blockThreads, warp);

                    if(static_cast<std::uint32_t>(status) != 0u)
                    {
                        if(warp.isLeader())
                            timeouts++;
                        break; // stop this consumer; the rest is counted below
                    }

                    auto const seq = out.sequenceNumber;
                    if(warp.isLeader())
                        claimedCount++;

                    // Every lane verifies its word from the producing warp's
                    // payload (a different warp). Cross-warp visibility check.
                    auto const word = payload[seq * wordsPerEntry + lane];
                    if(word != mix64(seq * 131u + lane))
                        wordErrors++;
                    if(out.request.firstId != seq || out.request.seed != mix64(seq * 131u))
                        wordErrors++;

                    if(warp.isLeader())
                    {
                        if(claimed[seq] != 0u)
                            duplicates++;
                        else
                            claimed[seq] = seq + 1u;
                    }
                }

                if(warp.isLeader())
                {
                    results[idx(consumerIndex)] = wordErrors;
                    results[idx(numProducers)] += claimedCount;
                    results[idx(numProducers + 2u)] += duplicates;
                    results[idx(numProducers + 5u)] += timeouts;
                }
            }
        }
    };

    void runCrossWarp(std::uint32_t numBlocks, std::uint64_t entriesPerBlock)
    {
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        REQUIRE(selector.getDeviceCount() > 0u);
        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();

        auto const numProducers = numBlocks / 2u;
        auto const totalItems = numProducers * entriesPerBlock;

        using Ring = graph::ReadyRing<int, 64u>;
        auto const ringBytes = Ring::storageBytes;
        auto const payloadBytes = totalItems * wordsPerEntry * sizeof(Word);
        auto const claimedBytes = totalItems * sizeof(std::uint64_t);
        auto const arrivalBytes = sizeof(Arrival);
        // results layout: numProducers per-consumer wordErrors, then:
        //  [np+0] total claimed, [np+1] total pushed, [np+2] total duplicates,
        //  [np+3] (unused), [np+4] (unused), [np+5] total timeouts,
        //  [np+6] total push failures
        auto const resultsCount = numProducers + 7u;

        auto hostOut = alpaka::onHost::allocHost<std::uint64_t>(resultsCount);
        std::memset(hostOut.data(), 0, resultsCount * sizeof(std::uint64_t));
        auto deviceOut = alpaka::onHost::allocLike(device, hostOut);

        auto hostPayload = alpaka::onHost::allocHost<std::uint8_t>(payloadBytes);
        std::memset(hostPayload.data(), 0, payloadBytes);
        auto devicePayload = alpaka::onHost::allocLike(device, hostPayload);

        auto hostClaimed = alpaka::onHost::allocHost<std::uint8_t>(claimedBytes);
        std::memset(hostClaimed.data(), 0, claimedBytes);
        auto deviceClaimed = alpaka::onHost::allocLike(device, hostClaimed);

        auto hostArrival = alpaka::onHost::allocHost<std::uint8_t>(arrivalBytes);
        std::memset(hostArrival.data(), 0, arrivalBytes);
        auto deviceArrival = alpaka::onHost::allocLike(device, hostArrival);

        auto hostRing = alpaka::onHost::allocHost<std::uint8_t>(ringBytes);
        std::memset(hostRing.data(), 0, ringBytes);
        auto deviceRing = alpaka::onHost::allocLike(device, hostRing);

        auto const threadSpec = alpaka::onHost::ThreadSpec{
            alpaka::Vec<std::uint32_t, 1u>{numBlocks},
            alpaka::Vec<std::uint32_t, 1u>{32u},
            alpaka::exec::gpuCuda};

        queue.enqueue(
            threadSpec,
            CrossWarpKernel{},
            deviceRing,
            devicePayload,
            deviceClaimed,
            deviceArrival,
            deviceOut,
            numBlocks,
            entriesPerBlock);
        alpaka::onHost::wait(queue);

        alpaka::onHost::memcpy(queue, hostOut, deviceOut);
        alpaka::onHost::wait(queue);

        auto const& r = hostOut;
        std::uint64_t totalWordErrors = 0u;
        for(std::size_t c = 0u; c < numProducers; ++c)
            totalWordErrors += r[c];

        INFO(
            "blocks=" << numBlocks << " entriesPerBlock=" << entriesPerBlock << " totalItems=" << totalItems
                      << " claimed=" << r[numProducers] << " pushed=" << r[numProducers + 1]
                      << " wordErrors=" << totalWordErrors << " duplicates=" << r[numProducers + 2]
                      << " timeouts=" << r[numProducers + 5] << " pushFails=" << r[numProducers + 6]);

        CHECK(r[numProducers + 5] == 0u); // no consumer timed out
        CHECK(r[numProducers + 6] == 0u); // no producer failed
        CHECK(r[numProducers + 1] == totalItems); // all pushed
        CHECK(r[numProducers] == totalItems); // all claimed
        CHECK(totalWordErrors == 0u); // every lane's payload word visible
        CHECK(r[numProducers + 2] == 0u); // no duplicates
    }
} // namespace

TEST_CASE("ready ring: cross-warp visibility, 8 blocks x 32 lanes, 200 items each", "[graph][ready_ring][gpu]")
{
    // 8 blocks: 4 producers, 4 consumers. 200 items per block = 800 total.
    runCrossWarp(8u, 200u);
}

TEST_CASE("ready ring: cross-warp visibility, 16 blocks x 32 lanes, 128 items each", "[graph][ready_ring][gpu]")
{
    // 16 blocks: 8 producers, 8 consumers. 128 items per block = 1024 total.
    runCrossWarp(16u, 128u);
}

#endif // GRAPH_HAS_CUDA
