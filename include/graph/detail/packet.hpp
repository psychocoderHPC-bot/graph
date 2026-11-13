// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/concepts.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Packet identity and storage description.
 *
 * A packet record is a small, trivially copyable value stored in device-visible
 * memory. It never owns storage: the packet pool owns the backing allocation
 * and hands out raw pointers. `PacketRecord::view()` is the single typed
 * reconstruction entry point used by the functor ABI.
 */
namespace graph
{
    /** Maximum number of ports a single packet can expose. */
    inline constexpr std::uint32_t maxPacketPorts = 4u;

    /** Fixed-capacity limits for the packet pool and its metadata.
     *
     * These are conservative starting values for the initial implementation.
     * A pool instantiation may use smaller capacities; peaks must stay below
     * these bounds so all metadata can live in preallocated storage.
     */
    struct PacketPoolTraits
    {
        /** Maximum logical elements in one packet. */
        static constexpr std::uint32_t maxPacketCapacity = 4096u;
        /** Maximum number of live packet slots. */
        static constexpr std::uint32_t maxPackets = 4096u;
        /** Maximum number of ready-queue entries. */
        static constexpr std::uint32_t maxReadyEntries = 4096u;
        /**
         * Maximum number of concurrent worker warps.
         *
         * Tied to the pool's tracking capacity (`poolMaxWorkers` in
         * `pool_control.hpp`), which sizes every per-worker array in the device
         * state. The 1024 value here was an unguarded constant below the legal
         * geometry (a 2048-worker A30 configuration), so it no longer shadows
         * the real limit: the real check is `validatePoolConfig` against
         * `poolMaxWorkers`, and this constant is not a second, stricter bound.
         */
        static constexpr std::uint32_t maxWorkers = 4096u;
        /** Maximum number of packets a single worker may reserve at once. */
        static constexpr std::uint32_t maxWorkerReservations = 8u;
        /** Maximum number of registered operations. */
        static constexpr std::uint32_t maxOperations = 32u;
        /** Maximum ports per packet (mirrors `maxPacketPorts`). */
        static constexpr std::uint32_t maxPorts = maxPacketPorts;
    };

    /** Typed multidimensional view over packet storage.
     *
     * Pitches are in bytes, matching `alpaka::MdSpan`.
     */
    template<typename T, std::size_t T_rank>
    using PacketMdSpan = alpaka::MdSpan<T, alpaka::Vec<std::size_t, T_rank>, alpaka::Vec<std::size_t, T_rank>>;

    /** Identity of a packet slot plus its reuse generation.
     *
     * `index` selects a physical slot. `generation` increments every time the
     * pool recycles the slot, so a stale handle that still names the index can
     * be detected instead of silently observing a different packet.
     */
    struct PacketHandle
    {
        std::uint32_t index{0u};
        std::uint32_t generation{0u};

        friend constexpr bool operator==(PacketHandle const&, PacketHandle const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<PacketHandle>);
    static_assert(std::is_standard_layout_v<PacketHandle>);

    /** Description of the payload currently stored in one packet slot. */
    struct PacketRecord
    {
        PacketHandle handle{};
        /** Logical number of valid elements (may be smaller than `capacity`). */
        std::uint32_t elementCount{0u};
        /** Physical element capacity of the slot. */
        std::uint32_t capacity{0u};
        ElementSchema schema{};

        /** Number of elements a consumer may touch: bounded by both fields. */
        ALPAKA_FN_HOST_ACC constexpr std::uint32_t logicalCount() const noexcept
        {
            return elementCount < capacity ? elementCount : capacity;
        }

        ALPAKA_FN_HOST_ACC constexpr bool empty() const noexcept
        {
            return logicalCount() == 0u;
        }

        /** True when the record describes a typed payload of rank `T_rank`. */
        template<std::size_t T_rank>
        ALPAKA_FN_HOST_ACC constexpr bool hasRank() const noexcept
        {
            return schema.rank == static_cast<std::uint32_t>(T_rank);
        }

        /** Rebuild a typed multidimensional view over `data`.
         *
         * This is the typed reconstruction entry point of the packet ABI. The
         * record's schema must have been validated on the host when the packet
         * was created. If the requested compile-time rank does not match the
         * record, the returned view has zero extents in every dimension, so any
         * access through it is out of bounds by construction and callers can
         * rely on the masked helpers in `views.hpp` to avoid touching memory.
         *
         * @param data base pointer of the packet storage; ownership stays with
         *             the packet pool.
         */
        template<typename T, std::size_t T_rank = 1u>
        ALPAKA_FN_HOST_ACC constexpr auto view(T* data) const noexcept
        {
            static_assert(T_rank > 0u);
            static_assert(T_rank <= maxSchemaRank);

            alpaka::Vec<std::size_t, T_rank> extents{};
            alpaka::Vec<std::size_t, T_rank> pitches{};
            if(schema.rank == static_cast<std::uint32_t>(T_rank))
            {
                for(std::size_t d = 0u; d < T_rank; ++d)
                {
                    extents[d] = schema.extents[d];
                    pitches[d] = schema.pitchBytes[d];
                }
            }
            return PacketMdSpan<T, T_rank>{data, extents, pitches};
        }
    };

    static_assert(std::is_trivially_copyable_v<PacketRecord>);
    static_assert(std::is_standard_layout_v<PacketRecord>);
} // namespace graph
