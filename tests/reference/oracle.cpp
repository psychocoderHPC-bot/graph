// SPDX-FileCopyrightText: 2025 René Widera
// SPDX-License-Identifier: MPL-2.0
//
// Scalar reference oracle for the particle-feedback example
// (graph-persistent-warp-pool-design.md, section 8).
//
// Worklist simulation of the exact graph:
//   Generate -> Filter -> (A|B) -> (feedback -> Filter | Sink)
// Random draws use only (seed, particleId, visit, purpose) via splitmix64 on
// an injective mixed counter (see oracle.hpp), so results are independent of
// chunk size, backend, worker, and order. Weight additions are checked:
// overflow is a hard error, never a wrap.
//
// Termination: the per-round survival probability is exactly 1/2, independent
// of the particle's property, so P(alive after k rounds) = 2^-k. The example
// terminates almost surely; every particle is destroyed exactly once. The
// visit guard (maxVisits) is a diagnostic safety net only: a particle that
// reaches it is reported via the destroyed mask and the sentinel weight
// kSinkSentinel, excluded from the histogram and FNV-1a checksum, and counted
// in nonTerminating. With a correct RNG, nonTerminating must be 0 at the
// default guard of 1000000 for all fixture sizes; a nonzero count is a bug
// signal, not expected behavior. See docs-rng-spec.md and test-vectors.md.
//
// Usage: oracle <N> <seed> [maxVisits]
#include "oracle.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace
{

    using namespace rng_oracle;

    struct Result
    {
        bool ok = false;
        std::string err;
        std::vector<Weight> finalWeights; // per id; kSinkSentinel if undestroyed
        std::vector<std::pair<Weight, uint64_t>> hist; // (weight,count), sorted
        uint64_t totalCount = 0; // destroyed count (histogram row sum)
        uint64_t nonTerminating = 0;
        uint64_t maxVisit = 0;
        uint64_t fnv1a = 0; // over destroyed weights, ID order
    };

    Result run(uint64_t n, Seed seed, uint64_t maxVisits)
    {
        Result r;
        std::vector<Particle> parts(n);
        std::vector<uint8_t> destroyed(n, 0);
        std::vector<Weight> sinkWeight(n, kSinkSentinel);
        for(uint64_t i = 0; i < n; ++i)
        {
            parts[i] = Particle{i, draw32(seed, i, 0, Purpose::Property), kInitialWeight, 0};
        }

        std::vector<uint64_t> queue; // worklist of ids; initial = generation phase
        queue.reserve(n ? n : 1);
        for(uint64_t i = 0; i < n; ++i)
            queue.push_back(i);
        uint64_t maxVisit = 0;

        for(std::size_t head = 0; head < queue.size(); ++head)
        {
            uint64_t const id = queue[head];
            Particle& p = parts[id];

            // Visit guard: a particle that exceeds maxVisits is reported as
            // non-terminating and dropped from the worklist. The guard is a
            // diagnostic safety net, not a semantic boundary: with a correct
            // uniform RNG the per-round survival probability is exactly 1/2, so
            // P(alive after k rounds) = 2^-k and reaching the guard is a bug
            // signal (expected count N * 2^-maxVisits, i.e. 0 for every fixture).
            if(p.visit > maxVisits)
            {
                r.nonTerminating += 1;
                sinkWeight[id] = kSinkSentinel;
                continue;
            }
            maxVisit = std::max(maxVisit, p.visit);

            // FilterByProperty: fresh draw per visit, purpose FILTER.
            bool const bin1 = draw32(seed, id, p.visit, Purpose::Filter) >= p.property;

            // ProcessA / ProcessB: fresh decision draw, exact 0.5 threshold.
            Purpose const decision = bin1 ? Purpose::DecisionB : Purpose::DecisionA;
            bool const flag1 = draw32(seed, id, p.visit, decision) >= kHalfScale;

            if(flag1)
            {
                // Survives this round: increment weight, return to filter input.
                Weight const add = bin1 ? 1u : 1000u;
                if(p.weight > UINT32_MAX - add)
                {
                    r.err = "weight overflow (add " + std::to_string(add) + ") for particle " + std::to_string(id);
                    return r;
                }
                p.weight += add;
                p.visit += 1; // visit increments on each feedback return
                queue.push_back(id);
            }
            else
            {
                // DestroyParticles: write final weight exactly once.
                if(destroyed[id])
                {
                    r.err = "duplicate sink for particle " + std::to_string(id);
                    return r;
                }
                destroyed[id] = 1;
                sinkWeight[id] = p.weight;
            }
        }

        r.ok = true;
        r.finalWeights = std::move(sinkWeight);
        r.maxVisit = maxVisit;
        r.totalCount = n - r.nonTerminating;

        // Sparse histogram of destroyed final weights, sorted ascending.
        // Use a map so equal weights merge regardless of ID order.
        std::map<Weight, uint64_t> h;
        for(Weight w : r.finalWeights)
            if(w != kSinkSentinel)
                h[w]++;
        for(auto& kv : h)
            r.hist.push_back({kv.first, kv.second});

        // FNV-1a 64 over destroyed final weights in ID order (sentinel excluded).
        r.fnv1a = 1'469'598'103'934'665'603ULL;
        for(Weight w : r.finalWeights)
        {
            if(w == kSinkSentinel)
                continue;
            r.fnv1a ^= static_cast<uint64_t>(w);
            r.fnv1a *= 1'099'511'628'211ULL;
        }
        return r;
    }

} // namespace

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::fprintf(stderr, "usage: %s <N> <seed> [maxVisits]\n", argv[0]);
        return 2;
    }
    uint64_t const n = std::strtoull(argv[1], nullptr, 10);
    uint64_t const seed = std::strtoull(argv[2], nullptr, 10);
    uint64_t const maxVisits = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : 1'000'000;

    Result const r = run(n, seed, maxVisits);
    if(!r.ok)
    {
        std::fprintf(stderr, "ERROR: %s\n", r.err.c_str());
        return 1;
    }

    // Termination check: with a correct uniform RNG, P(alive after k rounds)
    // = 2^-k, so at the default guard of 1000000 no particle may survive.
    // A nonzero count is a bug signal (wrong RNG or wrong model), not
    // expected behavior.
    if(r.nonTerminating > 0)
        std::fprintf(
            stderr,
            "WARNING: %" PRIu64 " particle(s) reached the visit guard; per-round "
            "survival probability must be exactly 1/2, so this "
            "indicates a defective RNG or model, not a property of "
            "the design.\n",
            r.nonTerminating);

    std::printf("N=%" PRIu64 " seed=%" PRIu64 " maxVisits=%" PRIu64 "\n", n, seed, maxVisits);
    std::printf("histogram: weight count\n");
    for(auto const& wc : r.hist)
        std::printf("%" PRIu32 " %" PRIu64 "\n", wc.first, wc.second);
    std::printf("total %" PRIu64 "\n", r.totalCount);
    std::printf("nonTerminating %" PRIu64 "\n", r.nonTerminating);
    std::printf("maxVisit %" PRIu64 "\n", r.maxVisit);
    std::printf("fnv1a %016" PRIX64 "\n", r.fnv1a);
    return 0;
}
