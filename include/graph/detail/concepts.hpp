// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Shared, accelerator-agnostic type vocabulary for the graph worker pool.
 *
 * This header intentionally contains descriptions only. It defines what an
 * element is, how a typed view is described, and which alpaka view types the
 * functor ABI accepts. It contains no scheduler logic, no ring code, and no
 * particle algorithms.
 *
 * All schema construction and validation helpers are host-only
 * (`ALPAKA_FN_HOST`) because schemas are built and checked on the host while
 * the graph is assembled. The resulting `ElementSchema` is a plain trivially
 * copyable value that later travels to the device unchanged.
 */
namespace graph
{
    /** Maximum number of dimensions a packet schema can describe. */
    inline constexpr std::uint32_t maxSchemaRank = 4u;

    /** Stable (per build) identifier of an element type.
     *
     * The value is derived from the compiler's type signature via FNV-1a, so
     * two schemas compare equal exactly when they were produced for the same
     * element type in the same build. It is not a cross-compiler or
     * cross-language ABI token.
     */
    using ElementTypeId = std::uint64_t;

    /** Sentinel for a schema that does not describe a typed payload (empty port). */
    inline constexpr ElementTypeId invalidElementTypeId = 0u;

    namespace detail
    {
        constexpr std::uint64_t fnv1a64(char const* text) noexcept
        {
            std::uint64_t hash = 14'695'981'039'346'656'037ull;
            while(*text != '\0')
            {
                hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(*text));
                hash *= 1'099'511'628'211ull;
                ++text;
            }
            return hash;
        }

        template<typename T>
        constexpr char const* typeSignature() noexcept
        {
#if defined(_MSC_VER)
            return __FUNCSIG__;
#else
            return __PRETTY_FUNCTION__;
#endif
        }
    } // namespace detail

    /** Identifier of the element type `T` for schema compatibility checks. */
    template<typename T>
    ALPAKA_FN_HOST constexpr ElementTypeId elementTypeId() noexcept
    {
        return detail::fnv1a64(detail::typeSignature<std::remove_cv_t<T>>());
    }

    /** Element types that may travel through the packet pool.
     *
     * Packet elements are copied bitwise between packets and across the
     * host/device boundary, so they must be trivially copyable and have a
     * stable layout.
     */
    template<typename T>
    concept PacketElement
        = std::is_trivially_copyable_v<std::remove_cv_t<T>> && std::is_standard_layout_v<std::remove_cv_t<T>>;

    /** An `alpaka::MdSpan`-like view whose element type may be a packet element. */
    template<typename T>
    concept ElementView
        = alpaka::concepts::IMdSpan<T> && PacketElement<typename std::remove_reference_t<T>::value_type>;

    /** Read-only view accepted by a functor input port. */
    template<typename T>
    concept InputPortView
        = alpaka::concepts::IDataSource<T> && PacketElement<typename std::remove_reference_t<T>::value_type>;

    /** Mutable view accepted by a functor output port. */
    template<typename T>
    concept OutputPortView = ElementView<T>;

    /** Particle view alias kept for readability in functor signatures. */
    template<typename T>
    concept ParticleViewLike = ElementView<T>;

    /** Minimal collection abstraction for a group of typed port views.
     *
     * A collection exposes its number of views and indexed access to them. It
     * is intentionally weaker than a container so a fixed-size array or a
     * view tuple can model an output-port bundle.
     */
    template<typename T>
    concept ViewCollection = requires(T const& collection, std::size_t index) {
        { collection.size() } -> std::convertible_to<std::size_t>;
        collection[index];
    };

    /** POD description of one typed packet payload.
     *
     * The descriptor carries enough information to
     * - reject an edge whose producer and consumer disagree on the element type
     *   or shape,
     * - rebuild an `alpaka::MdSpan` from a raw device pointer, and
     * - check that a logical element count fits the physical capacity.
     *
     * Pitches are stored in bytes and follow the alpaka convention: the last
     * dimension is contiguous (`pitchBytes[rank - 1] == sizeof(T)`).
     */
    struct ElementSchema
    {
        ElementTypeId elementType{invalidElementTypeId};
        std::uint32_t rank{0u};
        std::uint32_t elementSize{0u};
        std::uint32_t alignment{0u};
        std::array<std::size_t, maxSchemaRank> extents{};
        std::array<std::size_t, maxSchemaRank> pitchBytes{};

        friend constexpr bool operator==(ElementSchema const&, ElementSchema const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<ElementSchema>);
    static_assert(std::is_standard_layout_v<ElementSchema>);

    /** Build a contiguous row-major schema for `T` from `extents`. */
    template<typename T, alpaka::concepts::Vector T_Extents>
    ALPAKA_FN_HOST constexpr ElementSchema makeElementSchema(T_Extents const& extents) noexcept
    {
        constexpr std::size_t rank = static_cast<std::size_t>(T_Extents::dim());
        static_assert(rank > 0u);
        static_assert(rank <= maxSchemaRank);
        static_assert(PacketElement<T>);

        auto const pitchBytes = alpaka::calculatePitchesFromExtents<T>(extents);
        ElementSchema schema{};
        schema.elementType = elementTypeId<T>();
        schema.rank = static_cast<std::uint32_t>(rank);
        schema.elementSize = static_cast<std::uint32_t>(sizeof(T));
        schema.alignment = static_cast<std::uint32_t>(alignof(T));
        for(std::size_t d = 0u; d < rank; ++d)
        {
            schema.extents[d] = static_cast<std::size_t>(extents[d]);
            schema.pitchBytes[d] = static_cast<std::size_t>(pitchBytes[d]);
        }
        return schema;
    }

    /** Build a schema for `T` from explicit extents and byte pitches. */
    template<typename T, alpaka::concepts::Vector T_Extents, alpaka::concepts::Vector T_Pitches>
    ALPAKA_FN_HOST constexpr ElementSchema makeElementSchema(
        T_Extents const& extents,
        T_Pitches const& pitchBytes) noexcept
    {
        constexpr std::size_t rank = static_cast<std::size_t>(T_Extents::dim());
        static_assert(T_Pitches::dim() == T_Extents::dim());
        static_assert(rank > 0u);
        static_assert(rank <= maxSchemaRank);
        static_assert(PacketElement<T>);

        ElementSchema schema{};
        schema.elementType = elementTypeId<T>();
        schema.rank = static_cast<std::uint32_t>(rank);
        schema.elementSize = static_cast<std::uint32_t>(sizeof(T));
        schema.alignment = static_cast<std::uint32_t>(alignof(T));
        for(std::size_t d = 0u; d < rank; ++d)
        {
            schema.extents[d] = static_cast<std::size_t>(extents[d]);
            schema.pitchBytes[d] = static_cast<std::size_t>(pitchBytes[d]);
        }
        return schema;
    }

    /** Validate a schema before it is stored in an operation table.
     *
     * `rank == 0` describes an absent port (a source without an input or a sink
     * without an output) and is valid. Otherwise the dimensions up to `rank` must
     * be populated and the unused tail must be zero.
     */
    ALPAKA_FN_HOST constexpr bool isValid(ElementSchema const& schema) noexcept
    {
        if(schema.rank > maxSchemaRank)
            return false;
        if(schema.rank == 0u)
            return schema.elementType == invalidElementTypeId && schema.elementSize == 0u;
        if(schema.elementType == invalidElementTypeId)
            return false;
        if(schema.elementSize == 0u)
            return false;
        // Alignment must be a power of two, as required by the language.
        if(schema.alignment == 0u || (schema.alignment & (schema.alignment - 1u)) != 0u)
            return false;
        for(std::size_t d = 0u; d < schema.rank; ++d)
        {
            if(schema.extents[d] == 0u || schema.pitchBytes[d] == 0u)
                return false;
        }
        for(std::size_t d = schema.rank; d < maxSchemaRank; ++d)
        {
            if(schema.extents[d] != 0u || schema.pitchBytes[d] != 0u)
                return false;
        }
        return true;
    }

    /** Exact schema equality: same element type, shape, pitch, and alignment. */
    ALPAKA_FN_HOST constexpr bool schemasCompatible(ElementSchema const& lhs, ElementSchema const& rhs) noexcept
    {
        if(lhs.elementType != rhs.elementType)
            return false;
        if(lhs.rank != rhs.rank || lhs.elementSize != rhs.elementSize || lhs.alignment != rhs.alignment)
            return false;
        for(std::size_t d = 0u; d < maxSchemaRank; ++d)
        {
            if(lhs.extents[d] != rhs.extents[d] || lhs.pitchBytes[d] != rhs.pitchBytes[d])
                return false;
        }
        return true;
    }

    /** Edge compatibility: the destination may consume a sub-range of the source.
     *
     * Producer and consumer must agree on element type and rank. Every
     * destination extent must fit inside the corresponding source extent; the
     * destination element alignment must not exceed the source alignment.
     */
    ALPAKA_FN_HOST constexpr bool schemasEdgeCompatible(
        ElementSchema const& source,
        ElementSchema const& destination) noexcept
    {
        if(source.elementType != destination.elementType)
            return false;
        if(source.rank != destination.rank || source.elementSize != destination.elementSize)
            return false;
        if(destination.rank == 0u)
            return true;
        if(destination.alignment > source.alignment)
            return false;
        for(std::size_t d = 0u; d < destination.rank; ++d)
        {
            if(destination.extents[d] > source.extents[d])
                return false;
        }
        return true;
    }
} // namespace graph
