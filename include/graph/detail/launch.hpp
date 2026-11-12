// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/config.hpp>

#if GRAPH_HAS_CUDA
#    include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

/** @file
 *
 * Launch geometry for the persistent worker pool.
 *
 * The pool is one warp per block during bring-up. A worker is one hardware warp,
 * so the launch geometry is
 *
 *     blocks  = visible_multiprocessors * resident_blocks_per_multiprocessor
 *     workers = blocks * warps_per_block
 *
 * and the policy picks the legal candidate with the most resident workers.
 *
 * Co-residency is established by occupancy arithmetic over queried or measured
 * limits **plus** the startup rendezvous test. It is not a hardware guarantee:
 * alpaka has no occupancy API and no cooperative launch, and an ordinary launch
 * of an occupancy-sized grid does not promise that every block is resident at
 * the same time. The rendezvous test in `tests/integration/test_residency.cpp`
 * is the empirical gate that turns the arithmetic into evidence.
 *
 * Numbers used here:
 * - Queried through alpaka device properties: visible multiprocessors, warp
 *   width, maximum threads per block.
 * - Queried through the narrow backend adapter below: maximum resident blocks
 *   per multiprocessor, registers per multiprocessor, maximum resident threads
 *   per multiprocessor, shared memory per multiprocessor.
 * - Captured at build time: registers per thread of the persistent kernel
 *   (`GRAPH_KERNEL_REGISTERS`, see `cmake/GraphRegisters.cmake`). Registers per
 *   thread cannot be queried without a kernel handle.
 * - Measured on the HAL A30 (P1-PROTOCOL-PROBES): register allocation
 *   granularity of 8 registers per thread, four register sub-partitions per SM,
 *   and the block-per-SM model `4 * floor(16384 / (regs * warpSize))`. The same
 *   measurements give the 32 blocks/SM and 64 warps/SM caps for one-warp and
 *   multi-warp blocks.
 * - Conservative assumption: when no capture exists the fallback register count
 *   from `config.hpp` is used. It can lower the wave, never raise it.
 *
 * No accelerator is selected with the preprocessor. Backend differences are
 * confined to `graph::detail::{cuda,cpu}` and dispatched on the alpaka API tag.
 */
namespace graph
{
    /** Device limits needed to size the persistent launch.
     *
     * `registersPerThreadHint` is the per-thread allocation of the compiled
     * persistent kernel. `sharedMemBytesPerBlock` is the dynamic shared memory
     * the kernel requests (zero for the bring-up kernel).
     */
    struct DeviceLaunchProperties
    {
        int multiprocessors{0};
        int maxThreadsPerBlock{0};
        int maxBlocksPerMultiprocessor{0};
        int warpSize{0};
        int registersPerThreadHint{0};
        int sharedMemBytesPerBlock{0};
    };

    /** Per-multiprocessor resources used by the occupancy model.
     *
     * The defaults are the measured A30 (sm_80) values. The CUDA adapter
     * replaces the queryable fields; the register sub-partition model is the
     * measured/inferred part and therefore stays documented here.
     */
    struct OccupancyLimits
    {
        /** 32-bit registers per multiprocessor. Queried on CUDA. */
        std::uint32_t registersPerMultiprocessor{65536u};
        /** Registers are allocated in multiples of this value (measured A30). */
        std::uint32_t registerAllocationGranularity{8u};
        /** Register sub-partitions per multiprocessor (measured A30). */
        std::uint32_t subpartitionsPerMultiprocessor{4u};
        /** Maximum resident threads per multiprocessor. Queried on CUDA. */
        std::uint32_t maxThreadsPerMultiprocessor{2048u};
        /** Maximum resident warps per multiprocessor. Derived from threads. */
        std::uint32_t maxWarpsPerMultiprocessor{64u};
        /** Shared memory per multiprocessor in bytes. Queried on CUDA. */
        std::uint32_t sharedMemBytesPerMultiprocessor{167936u};
    };

    /** Warp-count policy for the persistent launch.
     *
     * The default is one warp per block, which needs no warp-local barrier:
     * block-wide synchronization is sufficient. Multi-warp blocks require both
     * `allowMultiWarpBlocks` and the compile-time option
     * `GRAPH_ENABLE_MULTI_WARP_BLOCKS`. See `computeLaunchGeometry()` for the
     * refusal.
     */
    struct LaunchPolicy
    {
        std::uint32_t maxWarpsPerBlock{1u};
        bool allowMultiWarpBlocks{false};
    };

    /** Concrete launch shape and the occupancy number behind it. */
    struct LaunchGeometry
    {
        std::uint32_t blocks{0};
        std::uint32_t threadsPerBlock{0};
        std::uint32_t warpsPerBlock{0};
        std::uint32_t workerCount{0};
        std::uint32_t residentsPerSm{0};
        std::uint32_t smCount{0};
    };

    /** Result of the one-wave check. */
    struct LaunchValidation
    {
        bool singleWave{false};
        std::string reason;
    };

    /** Registers per thread used when no build-time capture is available. */
    inline constexpr int fallbackKernelRegisters = GRAPH_KERNEL_REGISTERS_FALLBACK;

    /** Registers per thread of the persistent kernel, captured or fallback. */
    inline constexpr int defaultKernelRegisters
        = GRAPH_KERNEL_REGISTERS > 0 ? GRAPH_KERNEL_REGISTERS : fallbackKernelRegisters;

    /** True when the register count came from the build-time capture. */
    inline constexpr bool kernelRegistersCaptured = GRAPH_KERNEL_REGISTERS > 0;

    namespace detail
    {
        template<typename... T>
        inline constexpr bool launchAlwaysFalse = false;

        ALPAKA_FN_HOST constexpr std::uint32_t roundUpToGranularity(
            std::uint32_t value,
            std::uint32_t granularity) noexcept
        {
            if(granularity <= 1u)
                return value;
            return ((value + granularity - 1u) / granularity) * granularity;
        }
    } // namespace detail

    /** Resident warps per multiprocessor limited by the register file.
     *
     * Implements the measured A30 per-sub-partition model
     *
     *     warpsPerSm = subpartitions * floor(registersPerSubpartition /
     *                                        (roundUp(regs, 8) * warpSize))
     *
     * This is deliberately *not* `floor(registersPerSm / (regs * warpSize))`:
     * the whole-SM formula over-predicts every measured non-power-of-two point.
     * The caller applies the warp, thread, block and shared-memory caps.
     *
     * @param registersPerThread per-thread register allocation of the kernel
     * @param warpSize hardware warp width
     * @param limits per-multiprocessor resources
     * @return non-negative warp count, zero for degenerate input
     */
    ALPAKA_FN_HOST inline std::uint32_t registerLimitedWarpsPerSm(
        std::uint32_t registersPerThread,
        std::uint32_t warpSize = 32u,
        OccupancyLimits const& limits = {}) noexcept
    {
        if(warpSize == 0u || limits.subpartitionsPerMultiprocessor == 0u || limits.registerAllocationGranularity == 0u
           || limits.registersPerMultiprocessor == 0u)
            return 0u;

        std::uint32_t const rounded
            = detail::roundUpToGranularity(std::max(registersPerThread, 1u), limits.registerAllocationGranularity);
        std::uint32_t const registersPerSubpartition
            = limits.registersPerMultiprocessor / limits.subpartitionsPerMultiprocessor;
        std::uint64_t const registersPerWarp = std::uint64_t{rounded} * std::uint64_t{warpSize};
        if(registersPerWarp == 0u)
            return 0u;

        std::uint64_t const warpsPerSubpartition = std::uint64_t{registersPerSubpartition} / registersPerWarp;
        std::uint64_t const warps = warpsPerSubpartition * std::uint64_t{limits.subpartitionsPerMultiprocessor};
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(warps, std::numeric_limits<std::uint32_t>::max()));
    }

    /** Resident blocks per multiprocessor for a given block shape.
     *
     * Takes the minimum of the block, warp, thread, register and shared-memory
     * limits. A zero return means no block of this shape fits.
     */
    ALPAKA_FN_HOST inline std::uint32_t residentBlocksPerMultiprocessor(
        DeviceLaunchProperties const& props,
        std::uint32_t warpsPerBlock,
        OccupancyLimits const& limits = {}) noexcept
    {
        if(warpsPerBlock == 0u || props.warpSize <= 0 || props.maxBlocksPerMultiprocessor <= 0
           || props.maxThreadsPerBlock <= 0)
            return 0u;

        std::uint64_t const threadsPerBlock
            = std::uint64_t{warpsPerBlock} * static_cast<std::uint64_t>(props.warpSize);
        if(threadsPerBlock > static_cast<std::uint64_t>(props.maxThreadsPerBlock))
            return 0u;

        if(limits.maxWarpsPerMultiprocessor == 0u || limits.maxThreadsPerMultiprocessor == 0u)
            return 0u;

        std::uint64_t limit = static_cast<std::uint64_t>(props.maxBlocksPerMultiprocessor);
        limit = std::min(limit, std::uint64_t{limits.maxWarpsPerMultiprocessor} / warpsPerBlock);
        limit = std::min(limit, std::uint64_t{limits.maxThreadsPerMultiprocessor} / threadsPerBlock);

        // A missing or invalid capture falls back to the conservative value so
        // an unset hint can never raise the computed wave.
        std::uint32_t const registers = props.registersPerThreadHint > 0
                                            ? static_cast<std::uint32_t>(props.registersPerThreadHint)
                                            : static_cast<std::uint32_t>(fallbackKernelRegisters);
        std::uint32_t const registerWarps
            = registerLimitedWarpsPerSm(registers, static_cast<std::uint32_t>(props.warpSize), limits);
        limit = std::min(limit, std::uint64_t{registerWarps} / warpsPerBlock);

        if(props.sharedMemBytesPerBlock > 0)
            limit = std::min(
                limit,
                std::uint64_t{limits.sharedMemBytesPerMultiprocessor}
                    / static_cast<std::uint64_t>(props.sharedMemBytesPerBlock));
        else if(props.sharedMemBytesPerBlock < 0)
            return 0u;

        return static_cast<std::uint32_t>(std::min<std::uint64_t>(limit, std::numeric_limits<std::uint32_t>::max()));
    }

    /** Build the geometry for exactly one block shape.
     *
     * No policy gate is applied here. Callers that accept a user warp-count
     * request must go through the `LaunchPolicy` overload below.
     *
     * @throws std::invalid_argument for a degenerate device description, a
     *         block that exceeds the device thread limit, or a shape that no
     *         multiprocessor can host.
     */
    ALPAKA_FN_HOST inline LaunchGeometry computeLaunchGeometry(
        DeviceLaunchProperties const& props,
        std::uint32_t warpsPerBlock,
        OccupancyLimits const& limits = {})
    {
        if(props.multiprocessors <= 0 || props.warpSize <= 0 || props.maxBlocksPerMultiprocessor <= 0)
            throw std::invalid_argument("graph: invalid device launch properties");
        if(warpsPerBlock == 0u)
            throw std::invalid_argument("graph: warps per block must be at least one");

        std::uint64_t const threadsPerBlock
            = std::uint64_t{warpsPerBlock} * static_cast<std::uint64_t>(props.warpSize);
        if(threadsPerBlock > static_cast<std::uint64_t>(props.maxThreadsPerBlock))
            throw std::invalid_argument("graph: block shape exceeds the device thread-per-block limit");

        std::uint32_t const residents = residentBlocksPerMultiprocessor(props, warpsPerBlock, limits);
        if(residents == 0u)
            throw std::invalid_argument("graph: no block of this shape fits on one multiprocessor");

        std::uint64_t const blocks = std::uint64_t{static_cast<std::uint32_t>(props.multiprocessors)} * residents;
        std::uint64_t const workers = blocks * warpsPerBlock;
        if(blocks > std::numeric_limits<std::uint32_t>::max() || workers > std::numeric_limits<std::uint32_t>::max()
           || threadsPerBlock > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("graph: launch geometry exceeds a 32-bit launch dimension");

        LaunchGeometry geometry;
        geometry.blocks = static_cast<std::uint32_t>(blocks);
        geometry.threadsPerBlock = static_cast<std::uint32_t>(threadsPerBlock);
        geometry.warpsPerBlock = warpsPerBlock;
        geometry.workerCount = static_cast<std::uint32_t>(workers);
        geometry.residentsPerSm = residents;
        geometry.smCount = static_cast<std::uint32_t>(props.multiprocessors);
        return geometry;
    }

    /** Choose the supported candidate with the most resident workers.
     *
     * One warp per block is always a candidate. Multi-warp candidates are only
     * evaluated when the caller sets `allowMultiWarpBlocks`; the request is then
     * refused unless the build defines `GRAPH_ENABLE_MULTI_WARP_BLOCKS`.
     *
     * The missing prerequisite for multi-warp blocks is a validated warp-local
     * barrier that orders the participating lanes' memory operations. A shuffle
     * only exchanges data, and a block-wide barrier must not appear in an
     * independently scheduled warp path. Until that adapter exists the option
     * stays off and the bring-up shape is one warp per block.
     *
     * Ties prefer fewer warps per block, because it needs less synchronization.
     *
     * @throws std::invalid_argument when a multi-warp request is not supported
     *         by the build, or when no legal candidate exists.
     */
    ALPAKA_FN_HOST inline LaunchGeometry computeLaunchGeometry(
        DeviceLaunchProperties const& props,
        LaunchPolicy const& policy = {},
        OccupancyLimits const& limits = {})
    {
        std::uint32_t const requestedMax
            = policy.allowMultiWarpBlocks ? std::max<std::uint32_t>(policy.maxWarpsPerBlock, 1u) : 1u;

        if(requestedMax > 1u && !static_cast<bool>(GRAPH_ENABLE_MULTI_WARP_BLOCKS))
            throw std::invalid_argument(
                "graph: multi-warp blocks are disabled; a validated warp-local barrier is the missing "
                "prerequisite for GRAPH_ENABLE_MULTI_WARP_BLOCKS");

        LaunchGeometry best{};
        bool haveCandidate = false;
        for(std::uint32_t warps = 1u; warps <= requestedMax; ++warps)
        {
            std::uint64_t const threadsPerBlock = std::uint64_t{warps} * static_cast<std::uint64_t>(props.warpSize);
            if(props.warpSize <= 0 || threadsPerBlock > static_cast<std::uint64_t>(props.maxThreadsPerBlock))
                continue;
            if(residentBlocksPerMultiprocessor(props, warps, limits) == 0u)
                continue;

            LaunchGeometry const candidate = computeLaunchGeometry(props, warps, limits);
            bool const better
                = candidate.workerCount > best.workerCount
                  || (candidate.workerCount == best.workerCount && candidate.warpsPerBlock < best.warpsPerBlock);
            if(!haveCandidate || better)
            {
                best = candidate;
                haveCandidate = true;
            }
        }

        if(!haveCandidate)
            throw std::invalid_argument("graph: no legal warp count per block for this device");
        return best;
    }

    /** Check that a launch fits into one resident wave.
     *
     * This re-derives the resident block count from @p props and @p limits and
     * compares the configured grid against it. A geometry that was built for a
     * different device, a block shape that disagrees with the warp width, or a
     * grid larger than `smCount * residentsPerSm` fails with a reason. Passing
     * this check is still arithmetic, not proof; the rendezvous test supplies
     * the empirical evidence.
     */
    ALPAKA_FN_HOST inline LaunchValidation validateSingleWave(
        LaunchGeometry const& geometry,
        DeviceLaunchProperties const& props,
        OccupancyLimits const& limits = {})
    {
        LaunchValidation result;
        if(geometry.blocks == 0u || geometry.workerCount == 0u || geometry.warpsPerBlock == 0u
           || geometry.threadsPerBlock == 0u)
        {
            result.reason = "geometry is empty";
            return result;
        }
        if(props.multiprocessors <= 0 || props.warpSize <= 0 || props.maxBlocksPerMultiprocessor <= 0)
        {
            result.reason = "invalid device launch properties";
            return result;
        }
        if(geometry.smCount != static_cast<std::uint32_t>(props.multiprocessors))
        {
            result.reason = "geometry was built for a different multiprocessor count";
            return result;
        }
        if(std::uint64_t{geometry.warpsPerBlock} * static_cast<std::uint64_t>(props.warpSize)
           != geometry.threadsPerBlock)
        {
            result.reason = "threads per block does not match warps per block times the warp width";
            return result;
        }
        if(geometry.threadsPerBlock > static_cast<std::uint32_t>(props.maxThreadsPerBlock))
        {
            result.reason = "threads per block exceeds the device limit";
            return result;
        }
        if(std::uint64_t{geometry.blocks} * geometry.warpsPerBlock != geometry.workerCount)
        {
            result.reason = "worker count does not match blocks times warps per block";
            return result;
        }

        std::uint32_t const atMostResident = residentBlocksPerMultiprocessor(props, geometry.warpsPerBlock, limits);
        if(atMostResident == 0u)
        {
            result.reason = "no block of this shape is resident per multiprocessor";
            return result;
        }
        if(geometry.residentsPerSm > atMostResident)
        {
            result.reason = "geometry claims more resident blocks per multiprocessor than the limits allow";
            return result;
        }

        std::uint64_t const waveCapacity = std::uint64_t{geometry.smCount} * atMostResident;
        if(std::uint64_t{geometry.blocks} > waveCapacity)
        {
            result.reason = "grid exceeds one resident wave: " + std::to_string(geometry.blocks) + " blocks > "
                            + std::to_string(waveCapacity) + " resident slots";
            return result;
        }

        result.singleWave = true;
        result.reason = "one resident wave";
        return result;
    }

    /** Relations returned by the backend adapter and the overview query. */
    struct LaunchEnvironment
    {
        DeviceLaunchProperties props;
        OccupancyLimits limits;
    };

    namespace detail
    {
        namespace cpu
        {
            /** Host policy: one single-thread block per visible core.
             *
             * The CPU executor schedules blocks across the process' threads. The
             * host device has no register file or fixed block residency, so the
             * adapter reports one resident block per core and the occupancy
             * limits are unused.
             */
            template<typename T_Device>
            ALPAKA_FN_HOST int maxBlocksPerMultiprocessor([[maybe_unused]] T_Device const& device) noexcept
            {
                return 1;
            }

            template<typename T_Device>
            ALPAKA_FN_HOST OccupancyLimits occupancyLimits([[maybe_unused]] T_Device const& device) noexcept
            {
                OccupancyLimits limits;
                limits.maxWarpsPerMultiprocessor = 1u;
                limits.maxThreadsPerMultiprocessor = 1u;
                return limits;
            }
        } // namespace cpu

        namespace cuda
        {
#if GRAPH_HAS_CUDA
            inline int deviceAttribute(int deviceIndex, cudaDeviceAttr attribute)
            {
                int value = 0;
                cudaError_t const status = cudaDeviceGetAttribute(&value, attribute, deviceIndex);
                if(status != cudaSuccess)
                    throw std::runtime_error(
                        std::string{"graph: cudaDeviceGetAttribute failed: "} + cudaGetErrorString(status));
                return value;
            }

            template<typename T_Device>
            ALPAKA_FN_HOST int maxBlocksPerMultiprocessor(T_Device const& device)
            {
                int const index = alpaka::onHost::getNativeHandle(device);
                return deviceAttribute(index, cudaDevAttrMaxBlocksPerMultiprocessor);
            }

            template<typename T_Device>
            ALPAKA_FN_HOST OccupancyLimits occupancyLimits(T_Device const& device)
            {
                int const index = alpaka::onHost::getNativeHandle(device);
                OccupancyLimits limits;
                int const warpSize = deviceAttribute(index, cudaDevAttrWarpSize);
                int const maxThreads = deviceAttribute(index, cudaDevAttrMaxThreadsPerMultiProcessor);
#    if CUDART_VERSION >= 11000
                int const maxRegisters = deviceAttribute(index, cudaDevAttrMaxRegistersPerMultiprocessor);
#    else
                int const maxRegisters = 65536;
#    endif
                int const maxShared = deviceAttribute(index, cudaDevAttrMaxSharedMemoryPerMultiprocessor);
                limits.registersPerMultiprocessor = static_cast<std::uint32_t>(std::max(maxRegisters, 0));
                limits.maxThreadsPerMultiprocessor = static_cast<std::uint32_t>(std::max(maxThreads, 0));
                limits.maxWarpsPerMultiprocessor
                    = warpSize > 0 ? limits.maxThreadsPerMultiprocessor / static_cast<std::uint32_t>(warpSize) : 0u;
                limits.sharedMemBytesPerMultiprocessor = static_cast<std::uint32_t>(std::max(maxShared, 0));
                return limits;
            }
#else
            template<typename T_Device>
            ALPAKA_FN_HOST int maxBlocksPerMultiprocessor([[maybe_unused]] T_Device const& device) noexcept
            {
                static_assert(launchAlwaysFalse<T_Device>, "graph: the CUDA backend is not enabled in this build");
                return 0;
            }

            template<typename T_Device>
            ALPAKA_FN_HOST OccupancyLimits occupancyLimits([[maybe_unused]] T_Device const& device) noexcept
            {
                static_assert(launchAlwaysFalse<T_Device>, "graph: the CUDA backend is not enabled in this build");
                return {};
            }
#endif
        } // namespace cuda
    } // namespace detail

    /** Read the device launch properties for @p device.
     *
     * Device properties come from alpaka. The maximum resident blocks per
     * multiprocessor is not exposed by alpaka, so the narrow backend adapter
     * supplies it. The host adapter reports one block per core, which makes the
     * generic formula produce the CPU policy without a special case.
     *
     * @param registersPerThreadHint capture of the persistent kernel; zero asks
     *        for `defaultKernelRegisters` (build-time capture or fallback)
     */
    template<typename T_Device>
    ALPAKA_FN_HOST DeviceLaunchProperties queryDeviceLaunchProperties(
        T_Device const& device,
        int registersPerThreadHint = 0)
    {
        auto const properties = device.getDeviceProperties();
        DeviceLaunchProperties props;
        props.multiprocessors = static_cast<int>(properties.multiProcessorCount);
        props.maxThreadsPerBlock = static_cast<int>(properties.maxThreadsPerBlock);
        props.warpSize = static_cast<int>(properties.warpSize);
        props.registersPerThreadHint
            = registersPerThreadHint > 0 ? registersPerThreadHint : graph::defaultKernelRegisters;
        // The bring-up kernel requests no dynamic shared memory.
        props.sharedMemBytesPerBlock = 0;

        using Api = ALPAKA_TYPEOF(device.getApi());
        if constexpr(std::is_same_v<std::remove_cvref_t<Api>, alpaka::api::Cuda>)
            props.maxBlocksPerMultiprocessor = detail::cuda::maxBlocksPerMultiprocessor(device);
        else if constexpr(std::is_same_v<std::remove_cvref_t<Api>, alpaka::api::Host>)
            props.maxBlocksPerMultiprocessor = detail::cpu::maxBlocksPerMultiprocessor(device);
        else
            static_assert(
                detail::launchAlwaysFalse<T_Device>,
                "graph: launch geometry supports the CPU and CUDA backends");
        return props;
    }

    /** Read the per-multiprocessor resources for @p device. */
    template<typename T_Device>
    ALPAKA_FN_HOST OccupancyLimits queryOccupancyLimits(T_Device const& device)
    {
        using Api = ALPAKA_TYPEOF(device.getApi());
        if constexpr(std::is_same_v<std::remove_cvref_t<Api>, alpaka::api::Cuda>)
            return detail::cuda::occupancyLimits(device);
        else if constexpr(std::is_same_v<std::remove_cvref_t<Api>, alpaka::api::Host>)
            return detail::cpu::occupancyLimits(device);
        else
            static_assert(
                detail::launchAlwaysFalse<T_Device>,
                "graph: launch geometry supports the CPU and CUDA backends");
    }

    /** Read everything needed to compute and validate a launch geometry. */
    template<typename T_Device>
    ALPAKA_FN_HOST LaunchEnvironment queryLaunchEnvironment(T_Device const& device, int registersPerThreadHint = 0)
    {
        return {queryDeviceLaunchProperties(device, registersPerThreadHint), queryOccupancyLimits(device)};
    }
} // namespace graph
