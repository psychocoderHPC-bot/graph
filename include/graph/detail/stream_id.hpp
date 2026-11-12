// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>

/** The one `graph::StreamId` definition shared by every graph header.
 *
 * `graph_model.hpp` uses `StreamId` to describe a logical stream in the graph
 * model, and `chunk_assembly.hpp` uses it to identify a destination stream in
 * chunk state. Both modules name the same concept, so they must see the same
 * type: a translation unit may include both headers, and a `StreamId` built by
 * one module has to be accepted by the other without a conversion.
 *
 * `StreamId` is deliberately not a bin id: two graph stages may use the same
 * numeric bin and still have distinct streams. It is not interchangeable with
 * `NodeId` either; both are separate strong types.
 */
namespace graph
{
    /** Strong identifier of a logical stream (a scheduled destination). */
    struct StreamId
    {
        using ValueType = std::uint32_t;

        static constexpr ValueType invalidValue = ~ValueType{0u};

        ValueType value{invalidValue};

        constexpr StreamId() noexcept = default;

        constexpr explicit StreamId(ValueType id) noexcept : value(id)
        {
        }

        static constexpr StreamId invalid() noexcept
        {
            return StreamId{};
        }

        ALPAKA_FN_HOST_ACC constexpr bool isValid() const noexcept
        {
            return value != invalidValue;
        }

        friend constexpr bool operator==(StreamId const&, StreamId const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<StreamId>);
    static_assert(std::is_standard_layout_v<StreamId>);
} // namespace graph
