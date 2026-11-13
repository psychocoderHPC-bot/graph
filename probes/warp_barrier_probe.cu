// SPDX-FileCopyrightText: 2026 René Widera
// SPDX-License-Identifier: MPL-2.0

/** Diagnostic probe for the ReadyRing root-cause (T7b-READY-RING-FIX).
 *
 * Question: the draft ring called `WarpGroup::all(true)` as a "warp rendezvous"
 * and claimed that, combined with the leader's release fence, it orders the
 * producer lanes' stores before the consumer reads them. That is the suspicion.
 * `all` is a warp *vote* (CUDA `__all_sync`): it observes predicates, it does not
 * order memory and it does not make one lane's stores visible to another.
 *
 * This probe runs one warp per block (32 threads). Lane 0 is the "consumer",
 * lanes 1..31 are "producers".
 *
 * Scenario A (vote-as-barrier, the draft protocol): each producer sets
 * `payload[lane-1] = lane-1` and then the warp calls `warp::all(acc, 1)` (a
 * vote, not a fence). Lane 0 then reads every producer's word and counts the
 * ones that are not yet the expected value.
 *
 * Scenario B (real barrier): same stores, but the warp calls
 * `onAcc::syncBlockThreads(acc)` (one warp per block, the sanctioned bring-up
 * barrier) before lane 0 reads.
 *
 * If A reports `voteInvisible > 0` and B reports `barrierInvisible == 0`, the
 * vote is not a memory barrier and `syncBlockThreads` is, which is the root
 * cause of the draft's failure and the reason a real `WarpBarrier` is required.
 *
 * Only built for the CUDA configuration.
 */

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <iostream>

namespace
{
    struct ProbeKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto payload,
            alpaka::concepts::IMdSpan auto out) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            auto const width = alpaka::onAcc::warp::getSize(acc);
            auto const lane = alpaka::onAcc::warp::getLaneIdx(acc);
            bool const isLeader = lane == 0u;

            if(width <= 1u)
            {
                // The CPU/host executor has a width-one warp; it cannot
                // demonstrate the GPU phenomenon, so it records width and
                // does not contribute.
                out[0] = width;
                out[1] = 0u;
                out[2] = 0u;
                out[3] = 0u;
                return;
            }

            // --- Scenario A: vote as barrier ---
            // Producers (lanes 1..31) set their word. This is a single shared
            // dependency, so a correct barrier must order the stores against
            // lane 0's later reads.
            if(!isLeader)
            {
                payload[lane - 1u] = static_cast<std::uint32_t>(lane - 1u);
            }

            // The draft "rendezvous": a warp vote, not a fence.
            alpaka::onAcc::warp::all(acc, 1);

            if(isLeader)
            {
                std::uint32_t invisible = 0u;
                for(std::uint32_t i = 0u; i + 1u < width; ++i)
                {
                    if(payload[i] != i)
                        ++invisible;
                }
                out[1] = invisible;
            }

            // --- Scenario B: real barrier ---
            if(!isLeader)
            {
                payload[lane - 1u] = 0u;
            }

            alpaka::onAcc::syncBlockThreads(acc);

            if(isLeader)
            {
                std::uint32_t invisible = 0u;
                for(std::uint32_t i = 0u; i + 1u < width; ++i)
                {
                    if(payload[i] != 0u)
                        ++invisible;
                }
                out[2] = invisible;
                out[3] = width;
            }
        }
    };
} // namespace

int main()
{
    auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable() || selector.getDeviceCount() == 0u)
    {
        std::cout << "no GPU available; probe skipped\n";
        return 0;
    }

    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto const threadSpec = alpaka::onHost::ThreadSpec{1u, 32u, alpaka::exec::gpuCuda};

    auto hostOut = alpaka::onHost::allocHost<std::uint32_t>(4);
    for(std::size_t i = 0u; i < 4u; ++i)
        hostOut[i] = ~std::uint32_t{0u};
    auto deviceOut = alpaka::onHost::allocLike(device, hostOut);

    auto hostPayload = alpaka::onHost::allocHost<std::uint32_t>(32u);
    for(std::size_t i = 0u; i < 32u; ++i)
        hostPayload[i] = ~std::uint32_t{0u};
    auto devicePayload = alpaka::onHost::allocLike(device, hostPayload);

    // Repeat many times: a missing fence is a race, so a single pass can pass
    // by luck. 200 launches make the vote-vs-barrier gap robust.
    std::uint32_t worstVote = 0u;
    std::uint32_t worstBarrier = 0u;
    for(int it = 0; it < 200; ++it)
    {
        queue.enqueue(threadSpec, ProbeKernel{}, devicePayload, deviceOut);
        alpaka::onHost::wait(queue);
        alpaka::onHost::memcpy(queue, hostOut, deviceOut);
        alpaka::onHost::wait(queue);
        if(hostOut[1] > worstVote)
            worstVote = hostOut[1];
        if(hostOut[2] > worstBarrier)
            worstBarrier = hostOut[2];
    }

    std::cout << "width=" << hostOut[3] << "  worstVoteInvisible=" << worstVote
              << "  worstBarrierInvisible=" << worstBarrier << "\n";
    std::cout << "CONCLUSION: ";
    if(worstVote > 0u && worstBarrier == 0u)
        std::cout << "warp::all(true) is a VOTE, not a memory barrier; "
                     "syncBlockThreads orders lanes. Use a real WarpBarrier.\n";
    else if(worstVote == 0u && worstBarrier == 0u)
        std::cout << "vote happened to order memory in this run (unreliable); "
                     "a vote still provides NO ordering guarantee.\n";
    else
        std::cout << "unexpected: vote=" << worstVote << " barrier=" << worstBarrier << " - investigate\n";
    return 0;
}
