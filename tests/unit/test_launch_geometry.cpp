// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_test_macros.hpp>
#include <graph/detail/launch.hpp>

#include <cstdint>
#include <stdexcept>

namespace
{
    /** A30-shaped properties: 56 SMs, 32 blocks/SM cap, warp width 32. */
    graph::DeviceLaunchProperties a30Properties(int registersPerThread)
    {
        graph::DeviceLaunchProperties props;
        props.multiprocessors = 56;
        props.maxThreadsPerBlock = 1024;
        props.maxBlocksPerMultiprocessor = 32;
        props.warpSize = 32;
        props.registersPerThreadHint = registersPerThread;
        props.sharedMemBytesPerBlock = 0;
        return props;
    }
} // namespace

TEST_CASE(
    "launch_geometry register-limited warps follow the measured A30 sub-partition rule",
    "[graph][launch][geometry]")
{
    // P1-PROTOCOL-PROBES section 2: measured points on the A30.
    struct Point
    {
        std::uint32_t registers;
        std::uint32_t warpsPerSm;
    };

    Point const points[] = {
        {48u, 40u},
        {56u, 36u},
        {64u, 32u},
        {72u, 28u},
        {80u, 24u},
        {82u, 20u},
        {88u, 20u},
        {96u, 20u},
        {112u, 16u},
        {128u, 16u},
    };

    for(Point const& point : points)
    {
        CAPTURE(point.registers, point.warpsPerSm);
        CHECK(graph::registerLimitedWarpsPerSm(point.registers) == point.warpsPerSm);
    }
}

TEST_CASE("launch_geometry resident blocks reproduce the measured A30 boundaries", "[graph][launch][geometry]")
{
    // The 48- and 56-register rows are clamped by the 32-blocks/SM cap.
    struct Point
    {
        int registers;
        std::uint32_t blocksPerSm;
    };

    Point const points[] = {
        {48, 32u},
        {56, 32u},
        {64, 32u},
        {72, 28u},
        {80, 24u},
        {82, 20u},
        {88, 20u},
        {96, 20u},
        {112, 16u},
        {128, 16u},
    };

    for(Point const& point : points)
    {
        CAPTURE(point.registers, point.blocksPerSm);
        CHECK(graph::residentBlocksPerMultiprocessor(a30Properties(point.registers), 1u) == point.blocksPerSm);
    }
}

TEST_CASE("launch_geometry thread, warp, block and shared-memory limits all bind", "[graph][launch][geometry]")
{
    SECTION("thread limit")
    {
        graph::OccupancyLimits limits;
        limits.maxThreadsPerMultiprocessor = 512u;
        // 512 threads / 32 = 16 blocks, below the register and block caps.
        CHECK(graph::residentBlocksPerMultiprocessor(a30Properties(20), 1u, limits) == 16u);
    }

    SECTION("warp limit with four warps per block")
    {
        // 64 warps/SM / 4 warps per block = 16 blocks; threads also give 16.
        CHECK(graph::residentBlocksPerMultiprocessor(a30Properties(20), 4u) == 16u);
    }

    SECTION("block limit")
    {
        auto props = a30Properties(20);
        props.maxBlocksPerMultiprocessor = 8;
        CHECK(graph::residentBlocksPerMultiprocessor(props, 1u) == 8u);
    }

    SECTION("shared-memory limit")
    {
        auto props = a30Properties(20);
        props.sharedMemBytesPerBlock = 8192;
        graph::OccupancyLimits limits;
        limits.sharedMemBytesPerMultiprocessor = 16384u;
        CHECK(graph::residentBlocksPerMultiprocessor(props, 1u, limits) == 2u);
    }

    SECTION("block shape larger than the device thread limit is rejected")
    {
        auto props = a30Properties(20);
        props.maxThreadsPerBlock = 64;
        CHECK(graph::residentBlocksPerMultiprocessor(props, 4u) == 0u);
    }
}

TEST_CASE("launch_geometry geometry picks the most resident workers", "[graph][launch][geometry]")
{
    auto const oneWarp = graph::computeLaunchGeometry(a30Properties(20), 1u);
    auto const twoWarps = graph::computeLaunchGeometry(a30Properties(20), 2u);

    CHECK(oneWarp.blocks == 56u * 32u);
    CHECK(oneWarp.workerCount == 56u * 32u);
    CHECK(twoWarps.blocks == 56u * 32u);
    CHECK(twoWarps.workerCount == 56u * 32u * 2u);
    CHECK(twoWarps.workerCount > oneWarp.workerCount);

    // The default policy is one warp per block and must stay there.
    auto const defaultPolicy = graph::computeLaunchGeometry(a30Properties(20), graph::LaunchPolicy{});
    CHECK(defaultPolicy.warpsPerBlock == 1u);
    CHECK(defaultPolicy.workerCount == oneWarp.workerCount);

    // A warp-count ceiling without the explicit opt-in does not enable them.
    graph::LaunchPolicy ceilingOnly;
    ceilingOnly.maxWarpsPerBlock = 4u;
    ceilingOnly.allowMultiWarpBlocks = false;
    CHECK(graph::computeLaunchGeometry(a30Properties(20), ceilingOnly).warpsPerBlock == 1u);
}

TEST_CASE("launch_geometry multi-warp blocks are refused without the compile-time flag", "[graph][launch][geometry]")
{
    graph::LaunchPolicy policy;
    policy.maxWarpsPerBlock = 2u;
    policy.allowMultiWarpBlocks = true;

#if GRAPH_ENABLE_MULTI_WARP_BLOCKS
    auto const geometry = graph::computeLaunchGeometry(a30Properties(20), policy);
    CHECK(geometry.warpsPerBlock == 2u);
    CHECK(geometry.workerCount == 56u * 32u * 2u);
#else
    CHECK_THROWS_AS(graph::computeLaunchGeometry(a30Properties(20), policy), std::invalid_argument);
#endif
}

TEST_CASE(
    "launch_geometry single-wave validation accepts the policy and rejects malformed shapes",
    "[graph][launch][geometry]")
{
    auto const props = a30Properties(20);
    auto const geometry = graph::computeLaunchGeometry(props, graph::LaunchPolicy{});
    CHECK(graph::validateSingleWave(geometry, props).singleWave);

    SECTION("grid larger than the wave")
    {
        auto oversized = geometry;
        oversized.blocks *= 2u;
        oversized.workerCount = oversized.blocks * oversized.warpsPerBlock;
        auto const validation = graph::validateSingleWave(oversized, props);
        CHECK_FALSE(validation.singleWave);
        CHECK_FALSE(validation.reason.empty());
    }

    SECTION("threads per block disagree with the warp width")
    {
        auto malformed = geometry;
        malformed.threadsPerBlock += 1u;
        CHECK_FALSE(graph::validateSingleWave(malformed, props).singleWave);
    }

    SECTION("multiprocessor count mismatch")
    {
        auto otherDevice = props;
        otherDevice.multiprocessors = 108;
        CHECK_FALSE(graph::validateSingleWave(geometry, otherDevice).singleWave);
    }

    SECTION("overstated resident blocks")
    {
        auto overstated = geometry;
        overstated.residentsPerSm = geometry.residentsPerSm + 1u;
        CHECK_FALSE(graph::validateSingleWave(overstated, props).singleWave);
    }

    SECTION("empty geometry")
    {
        CHECK_FALSE(graph::validateSingleWave(graph::LaunchGeometry{}, props).singleWave);
    }
}

TEST_CASE("launch_geometry host policy is one worker per visible core", "[graph][launch][geometry]")
{
    graph::DeviceLaunchProperties props;
    props.multiprocessors = 64;
    props.maxThreadsPerBlock = 4096;
    props.maxBlocksPerMultiprocessor = 1;
    props.warpSize = 1;
    props.registersPerThreadHint = graph::fallbackKernelRegisters;
    props.sharedMemBytesPerBlock = 0;

    auto const geometry = graph::computeLaunchGeometry(props, graph::LaunchPolicy{});
    CHECK(geometry.smCount == 64u);
    CHECK(geometry.residentsPerSm == 1u);
    CHECK(geometry.blocks == 64u);
    CHECK(geometry.threadsPerBlock == 1u);
    CHECK(geometry.warpsPerBlock == 1u);
    CHECK(geometry.workerCount == 64u);
    CHECK(graph::validateSingleWave(geometry, props).singleWave);
}
