// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <graph/detail/job.hpp>
#include <graph/detail/packet.hpp>
#include <graph/detail/views.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Functor ABI of the persistent worker pool.
 *
 * # Calling convention
 *
 * Every registered graph operation is invoked through exactly one function
 * pointer type per accelerator instantiation:
 *
 *     void (*)(T_Acc const& acc, JobDescriptor const& job, PacketAccessor const& accessor)
 *
 * - `acc` is the accelerator context of the invoking warp. All lanes of the
 *   warp execute the same call. The functor may query lane/warp information and
 *   use `WarpGroup` for warp-uniform values.
 * - `job` is an immutable descriptor. It is owned by the scheduler and stays
 *   valid for the duration of the call. It carries the packet handle, routing
 *   information, the request payload, and the sequence number. The device
 *   function pointer is resolved separately through `OperationTable`, so the
 *   descriptor is independent of the accelerator type.
 * - `accessor` resolves the typed input and output views for the packet. It
 *   borrows packet-pool storage; that storage stays valid for the duration of
 *   the call and is released by the scheduler afterwards.
 *
 * # What a functor may do
 *
 * - Read input views and write output views for the ports selected by
 *   `job.routing`.
 * - Iterate over the logical element count (for example
 *   `lane, lane + width, ... < count`) and use `WarpGroup` for ballots,
 *   broadcasts, and ranks.
 * - Use per-lane local state and warp-local exchanges.
 *
 * # What a functor must not do
 *
 * - Allocate memory, free memory, or touch host APIs.
 * - Use block-wide or grid-wide barriers; independent warps may be scheduled
 *   at different times. Warp-local synchronization only.
 * - Throw exceptions, call `std::function`, or rely on RTTI.
 * - Retain any pointer or view beyond the call.
 * - Assume it is invoked by lane 0 only; every lane of the warp enters the
 *   same call.
 * - Write outside the logical element range; use the masked helpers from
 *   `views.hpp` for partial packets.
 *
 * Functors return `void`. Error and completion reporting belongs to the
 * scheduler and the job status, not to the functor ABI.
 */
namespace graph
{
    /** Result of a host-side operation registration attempt. */
    enum class RegistrationResult : std::uint32_t
    {
        Ok = 0u,
        TableFull,
        DuplicateId,
        InvalidId,
        NullFunction,
        InvalidInputSchema,
        InvalidOutputSchema
    };

    /** Type-erased, resolved views for one packet invocation.
     *
     * The accessor stores pointers to packet records and to the packet storage.
     * Both are owned by the packet pool and remain valid for the duration of
     * one functor call. The accessor itself is trivially copyable and may be
     * constructed on the device from pool-owned metadata.
     *
     * Ports are independent. A missing port (null record) resolves to a
     * zero-extent view, so a source without an input or a sink without an
     * output can share the same ABI.
     */
    struct PacketAccessor
    {
        PacketRecord const* inputRecords[maxPacketPorts]{};
        void* inputData[maxPacketPorts]{};
        std::uint32_t inputPortCount{0u};

        PacketRecord const* outputRecords[maxPacketPorts]{};
        void* outputData[maxPacketPorts]{};
        std::uint32_t outputPortCount{0u};

        /** Attach an input port. Returns false when `port` is out of range. */
        ALPAKA_FN_HOST_ACC constexpr bool setInput(std::uint32_t port, PacketRecord const* record, void* data) noexcept
        {
            if(port >= maxPacketPorts)
                return false;
            inputRecords[port] = record;
            inputData[port] = data;
            if(port >= inputPortCount)
                inputPortCount = port + 1u;
            return true;
        }

        /** Attach an output port. Returns false when `port` is out of range. */
        ALPAKA_FN_HOST_ACC constexpr bool setOutput(
            std::uint32_t port,
            PacketRecord const* record,
            void* data) noexcept
        {
            if(port >= maxPacketPorts)
                return false;
            outputRecords[port] = record;
            outputData[port] = data;
            if(port >= outputPortCount)
                outputPortCount = port + 1u;
            return true;
        }

        /** Drop all port bindings. */
        ALPAKA_FN_HOST_ACC constexpr void clear() noexcept
        {
            for(std::uint32_t port = 0u; port < maxPacketPorts; ++port)
            {
                inputRecords[port] = nullptr;
                inputData[port] = nullptr;
                outputRecords[port] = nullptr;
                outputData[port] = nullptr;
            }
            inputPortCount = 0u;
            outputPortCount = 0u;
        }

        ALPAKA_FN_HOST_ACC constexpr PacketRecord const* inputRecord(std::uint32_t port) const noexcept
        {
            return port < inputPortCount ? inputRecords[port] : nullptr;
        }

        ALPAKA_FN_HOST_ACC constexpr PacketRecord const* outputRecord(std::uint32_t port) const noexcept
        {
            return port < outputPortCount ? outputRecords[port] : nullptr;
        }

        /** Read-only typed input view. Missing ports yield zero extents. */
        template<typename T, std::size_t T_rank = 1u>
        ALPAKA_FN_HOST_ACC constexpr auto inputView(std::uint32_t port = 0u) const noexcept
        {
            auto const* record = inputRecord(port);
            if(record != nullptr)
                return makePacketView<T const, T_rank>(*record, static_cast<T const*>(inputData[port]));
            return emptyView<T const, T_rank>(nullptr);
        }

        /** Mutable typed output view. Missing ports yield zero extents. */
        template<typename T, std::size_t T_rank = 1u>
        ALPAKA_FN_HOST_ACC constexpr auto outputView(std::uint32_t port = 0u) const noexcept
        {
            auto const* record = outputRecord(port);
            if(record != nullptr)
                return makePacketView<T, T_rank>(*record, static_cast<T*>(outputData[port]));
            return emptyView<T, T_rank>(nullptr);
        }

        /** Bounds-masked read-only span over the packet's logical elements. */
        template<typename T, std::size_t T_rank = 1u>
        ALPAKA_FN_HOST_ACC constexpr auto inputSpan(std::uint32_t port = 0u) const noexcept
        {
            auto const* record = inputRecord(port);
            if(record != nullptr)
                return makeCheckedPacketSpan<T const, T_rank>(*record, static_cast<T const*>(inputData[port]));
            return CheckedPacketSpan<T const, T_rank>{};
        }

        /** Bounds-masked mutable span over the packet's logical elements. */
        template<typename T, std::size_t T_rank = 1u>
        ALPAKA_FN_HOST_ACC constexpr auto outputSpan(std::uint32_t port = 0u) const noexcept
        {
            auto const* record = outputRecord(port);
            if(record != nullptr)
                return makeCheckedPacketSpan<T, T_rank>(*record, static_cast<T*>(outputData[port]));
            return CheckedPacketSpan<T, T_rank>{};
        }

    private:
        /** Zero-extent view used for absent ports. */
        template<typename T, std::size_t T_rank>
        ALPAKA_FN_HOST_ACC static constexpr auto emptyView(T* data) noexcept
        {
            PacketRecord empty{};
            empty.schema.rank = static_cast<std::uint32_t>(T_rank);
            empty.schema.elementSize = static_cast<std::uint32_t>(sizeof(std::remove_const_t<T>));
            return makePacketView<T, T_rank>(empty, data);
        }
    };

    static_assert(std::is_trivially_copyable_v<PacketAccessor>);
    static_assert(std::is_standard_layout_v<PacketAccessor>);

    /** One common functor ABI per accelerator instantiation. */
    template<typename T_Acc>
    using OperationFn = void (*)(T_Acc const&, JobDescriptor const&, PacketAccessor const&);

    /** Fixed-capacity operation registry and device-side resolver.
     *
     * The host registers operations: it validates id uniqueness and schemas and
     * stores the descriptor. The same table type is used on the device, where
     * `resolve()` maps the id from a job to a device function pointer. On CUDA
     * the table must be populated with device function addresses in device
     * code (a host `&functor` would not be a device address).
     *
     * The table is a plain aggregate so it can be placed in device global or
     * static memory without allocation.
     */
    template<typename T_Acc>
    struct OperationTable
    {
        static constexpr std::uint32_t maxOperations = PacketPoolTraits::maxOperations;

        struct Entry
        {
            OperationId id{};
            ElementSchema inputSchema{};
            ElementSchema outputSchema{};
            OperationFn<T_Acc> function{nullptr};
        };

        Entry entries[maxOperations]{};
        std::uint32_t count{0u};

        static_assert(std::is_pointer_v<OperationFn<T_Acc>>);
        static_assert(std::is_function_v<std::remove_pointer_t<OperationFn<T_Acc>>>);

        /** Validate and append one operation on the host. */
        ALPAKA_FN_HOST constexpr RegistrationResult registerOperation(
            OperationId id,
            ElementSchema const& inputSchema,
            ElementSchema const& outputSchema,
            OperationFn<T_Acc> function) noexcept
        {
            if(!id.isValid())
                return RegistrationResult::InvalidId;
            if(function == nullptr)
                return RegistrationResult::NullFunction;
            if(count >= maxOperations)
                return RegistrationResult::TableFull;
            if(!isValid(inputSchema))
                return RegistrationResult::InvalidInputSchema;
            if(!isValid(outputSchema))
                return RegistrationResult::InvalidOutputSchema;
            for(std::uint32_t i = 0u; i < count; ++i)
            {
                if(entries[i].id == id)
                    return RegistrationResult::DuplicateId;
            }
            entries[count] = Entry{id, inputSchema, outputSchema, function};
            ++count;
            return RegistrationResult::Ok;
        }

        /** Resolve an id to its function pointer; returns nullptr if unknown. */
        ALPAKA_FN_HOST_ACC constexpr OperationFn<T_Acc> resolve(OperationId id) const noexcept
        {
            for(std::uint32_t i = 0u; i < count; ++i)
            {
                if(entries[i].id == id)
                    return entries[i].function;
            }
            return nullptr;
        }

        /** Entry lookup used by the host graph builder for schema checks. */
        ALPAKA_FN_HOST_ACC constexpr Entry const* find(OperationId id) const noexcept
        {
            for(std::uint32_t i = 0u; i < count; ++i)
            {
                if(entries[i].id == id)
                    return &entries[i];
            }
            return nullptr;
        }

        /** Device-side binding of an already registered id.
         *
         * The worker pool uses this during its startup phase to replace host
         * addresses with device function addresses where the backend requires
         * it. Returns false for an unknown id.
         */
        ALPAKA_FN_ACC constexpr bool bindOperation(OperationId id, OperationFn<T_Acc> function) noexcept
        {
            for(std::uint32_t i = 0u; i < count; ++i)
            {
                if(entries[i].id == id)
                {
                    entries[i].function = function;
                    return true;
                }
            }
            return false;
        }
    };

    static_assert(std::is_trivially_copyable_v<OperationTable<void*>>);
    static_assert(std::is_standard_layout_v<OperationTable<void*>>);
} // namespace graph
