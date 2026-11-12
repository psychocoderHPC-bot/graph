// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/concepts.hpp>
#include <graph/detail/packet.hpp>

#include <cstddef>
#include <limits>
#include <type_traits>

/** Typed view construction and bounds-masking helpers.
 *
 * Input packets may hold fewer elements than their physical capacity (a partial
 * packet). Lanes iterate `lane, lane + width, ... < logicalCount`, so the
 * scheduler loop itself is already bounded. `CheckedPacketSpan` adds a second
 * line of defence: a masked read returns a default value and a masked write is
 * dropped when the linear index is outside the logical range, so a buggy or
 * partially masked loop can never touch storage past the valid elements.
 */
namespace graph
{
    /** Rebuild a typed view from a packet record and a raw pointer.
     *
     * The pointer must address the packet storage described by the record.
     */
    template<typename T, std::size_t T_rank = 1u>
    ALPAKA_FN_HOST_ACC constexpr auto makePacketView(PacketRecord const& record, T* data) noexcept
    {
        return record.template view<T, T_rank>(data);
    }

    /** Bounds-masking span over a sub-range of a typed packet view.
     *
     * `offset` and `size` are expressed in elements of `T` and describe a
     * logical range inside the flattened view. Element access through
     * `read()`/`write()` is masked against `size()`.
     */
    template<typename T, std::size_t T_rank>
    class CheckedPacketSpan
    {
    public:
        static_assert(T_rank > 0u);
        static_assert(T_rank <= maxSchemaRank);

        using View = PacketMdSpan<T, T_rank>;
        using ValueType = std::remove_const_t<T>;

        ALPAKA_FN_HOST_ACC constexpr CheckedPacketSpan() noexcept = default;

        ALPAKA_FN_HOST_ACC constexpr CheckedPacketSpan(View view, std::size_t offset, std::size_t size) noexcept
            : m_view(view)
            , m_offset(offset)
            , m_size(size)
        {
        }

        /** Underlying typed view. Its extents describe the full packet. */
        ALPAKA_FN_HOST_ACC constexpr View const& view() const noexcept
        {
            return m_view;
        }

        /** Number of logical elements in this span. */
        ALPAKA_FN_HOST_ACC constexpr std::size_t size() const noexcept
        {
            return m_size;
        }

        ALPAKA_FN_HOST_ACC constexpr bool empty() const noexcept
        {
            return m_size == 0u;
        }

        /** True when `index` names a valid logical element of this span. */
        ALPAKA_FN_HOST_ACC constexpr bool inRange(std::size_t index) const noexcept
        {
            return index < m_size;
        }

        /** Masked read. Out-of-range reads return a value-initialised element. */
        ALPAKA_FN_HOST_ACC constexpr ValueType read(std::size_t index) const noexcept
            requires(std::is_default_constructible_v<ValueType>)
        {
            if(!inRange(index))
                return ValueType{};
            return m_view[elementIndex(m_offset + index)];
        }

        /** Masked write. Out-of-range writes are dropped. */
        ALPAKA_FN_HOST_ACC constexpr void write(std::size_t index, ValueType value) const noexcept
            requires(!std::is_const_v<T>)
        {
            if(inRange(index))
                m_view[elementIndex(m_offset + index)] = value;
        }

        /** Narrow this span to a sub-range, clamped to the current range. */
        ALPAKA_FN_HOST_ACC constexpr CheckedPacketSpan subSpan(std::size_t offset, std::size_t size) const noexcept
        {
            auto const start = offset > m_size ? m_size : offset;
            auto const available = m_size - start;
            return CheckedPacketSpan{m_view, m_offset + start, size > available ? available : size};
        }

    private:
        /** Map a flattened logical index to a view index. */
        ALPAKA_FN_HOST_ACC constexpr auto elementIndex(std::size_t linear) const noexcept
        {
            if constexpr(T_rank == 1u)
            {
                return linear;
            }
            else
            {
                auto const extents = m_view.getExtents();
                alpaka::Vec<std::size_t, T_rank> index{};
                for(std::size_t d = T_rank; d-- > 0u;)
                {
                    auto const extent = static_cast<std::size_t>(extents[d]);
                    index[d] = extent != 0u ? linear % extent : 0u;
                    linear = extent != 0u ? linear / extent : 0u;
                }
                return index;
            }
        }

        // Mutable so `write()` can update through a const span handle while the
        // span itself keeps value semantics.
        mutable View m_view{};
        std::size_t m_offset{0u};
        std::size_t m_size{0u};
    };

    /** Build a checked span covering the packet's logical element range.
     *
     * `offset` is clamped to the logical count and `size` defaults to the
     * remaining logical elements. The resulting span never covers more than
     * `record.logicalCount()` elements, regardless of `capacity`.
     */
    template<typename T, std::size_t T_rank = 1u>
    ALPAKA_FN_HOST_ACC constexpr auto makeCheckedPacketSpan(
        PacketRecord const& record,
        T* data,
        std::size_t offset = 0u,
        std::size_t size = std::numeric_limits<std::size_t>::max()) noexcept
    {
        auto const logical = static_cast<std::size_t>(record.logicalCount());
        auto const start = offset > logical ? logical : offset;
        auto const available = logical - start;
        auto const count = size > available ? available : size;
        return CheckedPacketSpan<T, T_rank>{makePacketView<T, T_rank>(record, data), start, count};
    }
} // namespace graph
