// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <graph/detail/packet.hpp>

#include <cstdint>
#include <type_traits>

/** Host/device job descriptor.
 *
 * Everything in this header is a plain value type. A `JobDescriptor` is built
 * once (on the host for ingress jobs, or by a worker for continuations) and
 * copied bitwise into device-visible queue storage. It contains no pointers to
 * host objects, no owning handles, and no virtual dispatch. The device function
 * pointer is not part of the descriptor: it is resolved through
 * `OperationTable` so the descriptor stays independent of the accelerator
 * instantiation.
 */
namespace graph
{
    /** Strong identifier of a registered graph operation.
     *
     * `OperationId` is deliberately not implicitly convertible to an integer so
     * an id cannot be confused with a slot index, a packet index, or a port.
     * The invalid sentinel is produced by the default constructor.
     */
    struct OperationId
    {
        using ValueType = std::uint32_t;

        static constexpr ValueType invalidValue = ~ValueType{0u};

        ValueType value{invalidValue};

        constexpr OperationId() noexcept = default;

        constexpr explicit OperationId(ValueType id) noexcept : value(id)
        {
        }

        static constexpr OperationId invalid() noexcept
        {
            return OperationId{};
        }

        ALPAKA_FN_HOST_ACC constexpr bool isValid() const noexcept
        {
            return value != invalidValue;
        }

        friend constexpr bool operator==(OperationId const&, OperationId const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<OperationId>);
    static_assert(std::is_standard_layout_v<OperationId>);

    /** Kind of work carried by a job.
     *
     * `OPERATION` invokes a registered functor. The other kinds are control
     * traffic consumed by the scheduler or by a control stage; they never reach
     * a functor.
     */
    enum class JobKind : std::uint32_t
    {
        OPERATION = 0u,
        CONTROL,
        END_OF_INPUT,
        DRAIN,
        ABORT
    };

    /** True for a job that is handled by the scheduler instead of a functor. */
    ALPAKA_FN_HOST_ACC constexpr bool isControlJob(JobKind kind) noexcept
    {
        return kind != JobKind::OPERATION;
    }

    /** Ordered source request payload.
     *
     * `firstId` and `count` describe a resumable element-id range. `seed` keys
     * the deterministic random stream for generated elements.
     */
    struct RequestSpec
    {
        std::uint64_t firstId{0u};
        std::uint64_t count{0u};
        std::uint64_t seed{0u};

        friend constexpr bool operator==(RequestSpec const&, RequestSpec const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<RequestSpec>);
    static_assert(std::is_standard_layout_v<RequestSpec>);

    /** Input/output port selection for one functor invocation.
     *
     * `inputPort` selects the accessor input slot. `outputPort` is the first
     * output slot and `outputPortCount` is the number of consecutive slots the
     * functor may write. A sink uses `outputPortCount == 0`.
     */
    struct RoutingInfo
    {
        std::uint32_t inputPort{0u};
        std::uint32_t outputPort{0u};
        std::uint32_t outputPortCount{1u};

        friend constexpr bool operator==(RoutingInfo const&, RoutingInfo const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<RoutingInfo>);
    static_assert(std::is_standard_layout_v<RoutingInfo>);

    /** Trivially copyable work item observed by a worker warp.
     *
     * The descriptor is immutable for the lifetime of one invocation. The
     * packet handle names the input packet; the referenced storage is owned by
     * the packet pool and must stay valid until the functor returns.
     */
    struct JobDescriptor
    {
        OperationId operation{};
        JobKind kind{JobKind::OPERATION};
        PacketHandle packet{};
        RoutingInfo routing{};
        RequestSpec request{};
        std::uint64_t sequenceNumber{0u};

        friend constexpr bool operator==(JobDescriptor const&, JobDescriptor const&) = default;
    };

    static_assert(std::is_trivially_copyable_v<JobDescriptor>);
    static_assert(std::is_standard_layout_v<JobDescriptor>);
    static_assert(std::is_trivially_copyable_v<decltype(JobDescriptor::operation)>);
} // namespace graph
