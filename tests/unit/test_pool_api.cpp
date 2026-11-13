// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

// Host-only pool API checks. No accelerator kernel is launched here: the
// configuration validator and the detached backend exercise the host state
// machine, the submission status vocabulary, and the drain/abort result
// distinction without a worker kernel.

#include "toy_ops.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/pool_control.hpp>
#include <graph/pool.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>

namespace
{
    graph::PoolConfig makeToyConfig(std::uint32_t chunkCapacity)
    {
        graph::PoolConfig config;
        config.packetSchema
            = graph::makeElementSchema<graph::toy::ToyItem>(alpaka::Vec<std::size_t, 1u>{chunkCapacity});
        config.chunkCapacity = chunkCapacity;
        config.sourceOperation = graph::toy::defaultToyOperationIds.source;
        config.budget.liveParticles = 64u;
        config.budget.controlAllowance = 8u;
        config.maxImportBatch = 4u;
        return config;
    }

    graph::LaunchGeometry smallGeometry()
    {
        graph::LaunchGeometry geometry;
        geometry.blocks = 4u;
        geometry.threadsPerBlock = 32u;
        geometry.warpsPerBlock = 1u;
        geometry.workerCount = 4u;
        geometry.residentsPerSm = 4u;
        geometry.smCount = 1u;
        return geometry;
    }
} // namespace

TEST_CASE("pool config validation accepts a legal budget and names an undersized one", "[graph][pool][api]")
{
    auto const geometry = smallGeometry();
    auto config = makeToyConfig(8u);
    CHECK(graph::validatePoolConfig(config, geometry).empty());

    config.budget.liveParticles = 100000u;
    std::string const error = graph::validatePoolConfig(config, geometry);
    CHECK(!error.empty());
    CHECK(error.find("undersized") != std::string::npos);

    auto broken = makeToyConfig(8u);
    broken.packetSchema = graph::ElementSchema{};
    CHECK(!graph::validatePoolConfig(broken, geometry).empty());

    auto tooLarge = makeToyConfig(8u);
    tooLarge.chunkCapacity = 1024u;
    CHECK(!graph::validatePoolConfig(tooLarge, geometry).empty());

    auto badBatch = makeToyConfig(8u);
    badBatch.maxImportBatch = graph::poolStagingCapacity + 1u;
    CHECK(!graph::validatePoolConfig(badBatch, geometry).empty());
}

TEST_CASE("pool worker capacity is accommodated below the limit and refused by name above it", "[graph][pool][api]")
{
    // Every per-worker array in the device state is sized by
    // `graph::poolMaxWorkers`. The validator must either accommodate a geometry
    // that fits or refuse it with the named `workerCapacityExceeded` error; it
    // must never accept a geometry the kernel would then index out of range.
    auto config = makeToyConfig(8u);
    // A large slot budget keeps the capacity check the deciding one: the packet
    // and ready-ring formulas are satisfied for the biggest tracked geometry.
    constexpr std::uint32_t slots = 1u << 20u;

    // Above every compile-time worker default in the tree (the 896-block launch
    // wave and `PacketPoolTraits::maxWorkers == 1024`) the geometry is
    // accommodated, not clamped.
    auto large = smallGeometry();
    large.blocks = 2048u;
    large.workerCount = 2048u;
    CHECK(graph::validatePoolConfig(config, large, slots).empty());

    // At exactly `poolMaxWorkers` the worker-capacity check itself must not
    // fire; whatever else rejects the configuration names a different contract.
    auto atCapacity = smallGeometry();
    atCapacity.blocks = graph::poolMaxWorkers;
    atCapacity.workerCount = graph::poolMaxWorkers;
    CHECK(graph::validatePoolConfig(config, atCapacity, slots).find("workerCapacityExceeded") == std::string::npos);

    auto aboveCapacity = smallGeometry();
    aboveCapacity.blocks = graph::poolMaxWorkers + 1u;
    aboveCapacity.workerCount = graph::poolMaxWorkers + 1u;
    std::string const error = graph::validatePoolConfig(config, aboveCapacity, slots);
    CHECK(!error.empty());
    CHECK(error.find("workerCapacityExceeded") != std::string::npos);
    CHECK(error.find(std::to_string(graph::poolMaxWorkers + 1u)) != std::string::npos);

    // The budget is validated against the same capacity, not only the geometry.
    auto oversizedBudget = makeToyConfig(8u);
    oversizedBudget.budget.workers = graph::poolMaxWorkers + 1u;
    std::string const budgetError = graph::validatePoolConfig(oversizedBudget, smallGeometry(), slots);
    CHECK(!budgetError.empty());
    CHECK(budgetError.find("workerCapacityExceeded") != std::string::npos);

    // The named error bit is part of the public error vocabulary.
    CHECK(
        graph::poolErrorSet(
            graph::poolErrorBits(graph::PoolError::workerCapacityExceeded),
            graph::PoolError::workerCapacityExceeded));
}

TEST_CASE("pool that was never started reports never-started and rejects submissions", "[graph][pool][api]")
{
    graph::Pool pool;
    CHECK(!pool.started());

    std::array<graph::RequestSpec, 2u> requests{graph::RequestSpec{0u, 8u, 1u}, graph::RequestSpec{8u, 8u, 1u}};
    auto const submitted = pool.tryEnqueue(std::span<graph::RequestSpec const>{requests});
    CHECK(submitted.status == graph::PoolEnqueueStatus::closed);
    CHECK(submitted.accepted == 0u);
    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::closed);

    auto const result = pool.wait();
    CHECK(result.reason == graph::PoolStopReason::neverStarted);
    CHECK(!result.stopped());
}

TEST_CASE("detached pool accepts, closes, and rejects later submissions", "[graph][pool][api]")
{
    auto pool = graph::detail::makeDetachedPool(makeToyConfig(8u));
    CHECK(pool.started());

    std::array<graph::RequestSpec, 3u> requests{
        graph::RequestSpec{0u, 8u, 7u},
        graph::RequestSpec{8u, 8u, 7u},
        graph::RequestSpec{16u, 8u, 7u}};
    auto const submitted = pool.tryEnqueue(std::span<graph::RequestSpec const>{requests});
    CHECK(submitted.status == graph::PoolEnqueueStatus::accepted);
    CHECK(submitted.accepted == 3u);

    CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
    auto const late = pool.tryEnqueue(std::span<graph::RequestSpec const>{requests});
    CHECK(late.status == graph::PoolEnqueueStatus::closed);
    CHECK(late.accepted == 0u);
}

TEST_CASE("detached pool distinguishes drained from aborted", "[graph][pool][api]")
{
    SECTION("ordered close reports a drain")
    {
        auto pool = graph::detail::makeDetachedPool(makeToyConfig(8u));
        CHECK(pool.closeInput() == graph::PoolEnqueueStatus::accepted);
        auto const result = pool.wait();
        CHECK(result.reason == graph::PoolStopReason::stoppedByDrain);
        CHECK(result.stopped());
    }

    SECTION("abort reports an abort, not a drain")
    {
        auto pool = graph::detail::makeDetachedPool(makeToyConfig(8u));
        std::array<graph::RequestSpec, 1u> requests{graph::RequestSpec{0u, 512u, 3u}};
        CHECK(pool.tryEnqueue(std::span<graph::RequestSpec const>{requests}).accepted == 1u);
        pool.abort();
        auto const result = pool.wait();
        CHECK(result.reason == graph::PoolStopReason::stoppedByAbort);
        CHECK(graph::poolErrorSet(result.errorWord, graph::PoolError::aborted));
    }
}
