// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

/** Dedicated host<->device transfer path for the worker pool.
 *
 * # Why a separate queue
 *
 * The persistent worker kernel occupies the worker queue for the whole run.
 * Results and setup data must move while that kernel is resident, so every
 * transfer goes through `CopyQueue`, which owns a queue created separately from
 * the worker queue on the same device. The copy queue is never the queue that
 * runs the worker kernel.
 *
 * alpaka documents that two `Device::makeQueue()` calls are *not* formally
 * guaranteed to return independent backend queues (`Device.hpp`,
 * "you always get the same handle back … There is no guarantee that you will get
 * independent queues"). Independence is therefore an observed property, not an
 * API contract: `tests/integration/test_copy_queue_overlap.cpp` verifies on the
 * configured backend that a copy on the copy queue completes while the worker
 * queue is still busy. Correctness must never depend on the overlap; only
 * latency does.
 *
 * # Pinned host staging
 *
 * On CUDA/HIP a device->host copy is only asynchronous (and can only overlap the
 * worker queue) when the host buffer is pinned/mapped. `allocHost` returns
 * pageable memory; a copy into pageable memory blocks until the stream is
 * drained and silently destroys the overlap. The helpers below therefore reject
 * a host buffer that is not usable staging and tell the caller to use
 * `alpaka::onHost::allocMapped<T>(device, extents)` (or
 * `CopyQueue::allocateStaging<T>()`). The check rejects a host-api (pageable)
 * view and then runs the device accessibility query for a device-api view, so a
 * buffer the transfer device cannot reach also fails loudly. On a host backend
 * every host buffer is accessible and the check always passes.
 *
 * # Ordering contract
 *
 * The copy queue has *no* implicit ordering relationship with the worker queue.
 * A copy of kernel-produced data is only safe after the producing work is known
 * to be complete:
 *
 * - To copy data produced by an enqueue on the worker queue, first enqueue an
 *   event on the worker queue (it is signalled after that queue's preceding
 *   work), then call `copyQueue.orderAfter(event)` *before* enqueueing the copy.
 *   On the non-blocking copy queue `orderAfter` enqueues a stream dependency
 *   (`cudaStreamWaitEvent`/`hipStreamWaitEvent`); it does not block the host.
 *   Waiting for the event on the host (`token.wait()`) after the copy is *not*
 *   sufficient to order the copy itself.
 * - Alternatively, wait for the worker queue on the host before enqueueing the
 *   copy. That serializes the host and loses the overlap but is always correct.
 *
 * The copy queue only orders work enqueued on itself. Never assume that
 * enqueueing on the worker queue and then on the copy queue preserves the
 * program order across the two queues.
 *
 * # Buffer reuse
 *
 * Every helper returns a `CompletionToken`. A staging buffer must not be read
 * or overwritten before `token.wait()` returned or `token.ready()` reported
 * `true`. The token observes the event that is signalled after the copy on the
 * copy queue.
 *
 * # Allocation
 *
 * The copy helpers never allocate device memory. Device buffers are supplied by
 * the packet pool / caller. Only host staging buffers may be allocated through
 * `allocateStaging()`.
 */
namespace graph
{
    /** Pollable completion handle for one asynchronous copy.
     *
     * The token wraps an alpaka event that is enqueued on the copy queue after
     * the copy. `ready()` polls without blocking; `wait()` blocks until the
     * event is complete. Copying a token shares the same underlying event.
     */
    template<typename T_Event>
    class CompletionToken
    {
    public:
        using Event = T_Event;

        explicit CompletionToken(Event event) noexcept : m_event(std::move(event))
        {
        }

        /** Block until the observed copy has completed. */
        void wait() const
        {
            alpaka::onHost::wait(m_event);
        }

        /** Non-blocking poll: true once the observed copy has completed. */
        bool ready() const
        {
            return m_event.isComplete();
        }

        /** Access the underlying alpaka event (for example to enqueue a dependency). */
        Event const& event() const noexcept
        {
            return m_event;
        }

    private:
        Event m_event;
    };

    /** Owner of the dedicated transfer queue and its pinned host staging path.
     *
     * One instance is created per device next to the worker queue. The copy
     * queue is created by this class and is never used to launch worker
     * kernels.
     */
    template<typename T_Device>
    class CopyQueue
    {
    public:
        using Device = T_Device;
        using QueueType = decltype(std::declval<Device&>().makeQueue());
        using EventType = decltype(std::declval<QueueType const&>().makeEvent());
        using Token = CompletionToken<EventType>;

        /** Create the dedicated transfer queue on `device`.
         *
         * The queue uses alpaka's default policy (non-blocking, timing off), so
         * `enqueue` returns before the copy completes and completion is observed
         * through the returned token.
         */
        explicit CopyQueue(Device device) : m_device(std::move(device)), m_queue(m_device.makeQueue())
        {
        }

        CopyQueue(CopyQueue const&) = delete;
        CopyQueue& operator=(CopyQueue const&) = delete;
        CopyQueue(CopyQueue&&) noexcept = default;
        CopyQueue& operator=(CopyQueue&&) noexcept = default;

        /** Device that owns the transfer queue. */
        Device const& device() const noexcept
        {
            return m_device;
        }

        /** Transfer queue. Should only receive transfers and ordering events. */
        QueueType& queue() noexcept
        {
            return m_queue;
        }

        QueueType const& queue() const noexcept
        {
            return m_queue;
        }

        /** Allocate a pinned host staging buffer for `elementCount` elements.
         *
         * This is the only allocation performed by the copy path. Device-side
         * buffers stay owned by the pool and are never allocated here.
         */
        template<typename T>
        auto allocateStaging(std::size_t elementCount) const
        {
            return alpaka::onHost::allocMapped<T>(m_device, elementCount);
        }

        /** Enqueue a host->device copy and return its completion token.
         *
         * `source` must be a pinned/mapped host buffer. A pageable buffer (for
         * example from `onHost::allocHost`) throws `std::invalid_argument`
         * instead of silently blocking and losing the overlap.
         */
        template<typename T_Dest, typename T_Source>
        Token asyncCopyToDevice(T_Dest&& destination, T_Source const& source) const
        {
            requirePinnedHostBuffer(source, "host->device copy source");
            return enqueueCopy(std::forward<T_Dest>(destination), source);
        }

        /** Enqueue a device->host copy and return its completion token.
         *
         * `destination` must be a pinned/mapped host buffer. A pageable buffer
         * throws `std::invalid_argument` instead of silently blocking.
         */
        template<typename T_Dest, typename T_Source>
        Token asyncCopyToHost(T_Dest&& destination, T_Source const& source) const
        {
            requirePinnedHostBuffer(destination, "device->host copy destination");
            return enqueueCopy(std::forward<T_Dest>(destination), source);
        }

        /** Make the copy queue wait for `event` before running later copies.
         *
         * Use this to order a copy after work on another queue, typically an
         * event enqueued on the worker queue after the producing enqueue. The
         * copy queue does not order itself against any other queue. On the
         * non-blocking transfer queue this is a stream dependency and returns
         * immediately.
         */
        template<typename T_Event>
        void orderAfter(T_Event const& event) const
        {
            m_queue.waitFor(event);
        }

    private:
        template<typename T_Dest, typename T_Source>
        Token enqueueCopy(T_Dest&& destination, T_Source const& source) const
        {
            auto event = m_queue.makeEvent();
            alpaka::onHost::memcpy(m_queue, std::forward<T_Dest>(destination), source);
            m_queue.enqueue(event);
            return Token{std::move(event)};
        }

        template<typename T_View>
        void requirePinnedHostBuffer(T_View const& view, char const* what) const
        {
            if(!isAccessibleFromDevice(view))
            {
                throw std::invalid_argument(
                    std::string{"graph::CopyQueue: "} + what
                    + " is not a pinned/mapped host buffer. A pageable buffer makes the copy block until the "
                      "worker queue is drained and loses all overlap. Allocate the staging buffer with "
                      "alpaka::onHost::allocMapped<T>(device, extents) (for example via "
                      "CopyQueue::allocateStaging<T>()).");
            }
        }

        /** True when `view` is usable as pinned transfer staging for this device.
         *
         * Two conditions are checked:
         * 1. The view must have been created with the device's api. A pageable
         *    `onHost::allocHost` buffer carries the host api and is rejected
         *    without touching the backend, which avoids a spurious CUDA error
         *    for an unregistered host pointer. `onHost::allocMapped` creates the
         *    staging view with the device api.
         * 2. For a device-api view, the pointer must be device accessible
         *    (`onHost::isDataAccessible`, the same pointer-attribute query alpaka
         *    uses). This catches a hand-made view over memory the device cannot
         *    reach.
         *
         * On a host backend both allocations use the host api, so only the
         * accessibility query runs and every host buffer is accepted.
         */
        template<typename T_View>
        bool isAccessibleFromDevice(T_View const& view) const noexcept
        {
            using ViewApi = std::remove_cvref_t<decltype(alpaka::getApi(view))>;
            using DeviceApi = std::remove_cvref_t<decltype(alpaka::getApi(m_device))>;
            if constexpr(!std::same_as<ViewApi, DeviceApi>)
            {
                return false;
            }
            else
            {
                try
                {
                    return alpaka::onHost::isDataAccessible(m_device, view);
                }
                catch(...)
                {
                    return false;
                }
            }
        }

        Device m_device;
        QueueType m_queue;
    };
} // namespace graph
