// SPDX-FileCopyrightText: 2025 René Widera
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <graph/system_atomic.hpp>

#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#if GRAPH_HAS_CUDA
#    include <cuda/atomic>
#endif

namespace
{
    using Value = std::uint64_t;

#if GRAPH_HAS_CUDA
    /** Compile-time contract for the portable system-atomic alignment rule.
     *
     * `checkSupported<T>()` is the gate every production primitive calls. It
     * must accept both payload widths the graph uses and derive the natural
     * alignment for the type's size, with or without libcu++'s
     * `required_alignment` member.
     */
    template<typename T>
    constexpr bool systemAtomicAlignmentSatisfied()
    {
        graph::detail::cuda::detail::checkSupported<T>();
        return alignof(T) >= graph::detail::cuda::detail::systemAtomicRequiredAlignment<T>();
    }

    static_assert(systemAtomicAlignmentSatisfied<std::uint32_t>());
    static_assert(systemAtomicAlignmentSatisfied<std::uint64_t>());
    static_assert(graph::detail::cuda::detail::systemAtomicRequiredAlignment<std::uint32_t>() == 4u);
    static_assert(graph::detail::cuda::detail::systemAtomicRequiredAlignment<std::uint64_t>() == 8u);

    // Rejected-type example (kept commented out so this file compiles): an
    // 8-byte payload with only 4 bytes of alignment cannot satisfy the
    // 8-byte system-atomic requirement.
    //
    //   struct alignas(4) Misaligned64
    //   {
    //       unsigned char bytes[8];
    //   };
    //   static_assert(alignof(Misaligned64) >= sizeof(Misaligned64)); // fails: 4 < 8
#endif

    /** Mixing function shared by the producer and consumer so a reordered or
     *  duplicated slot is detected by the payload check, not only by the
     *  sequence number.
     */
    ALPAKA_FN_ACC inline constexpr Value mix64(Value value) noexcept
    {
        value ^= value >> 30u;
        value *= 0xbf58'476d'1ce4'e5b9ull;
        value ^= value >> 27u;
        value *= 0x94d0'49bb'1331'11ebull;
        value ^= value >> 31u;
        return value;
    }

    auto makeTestDevice()
    {
#if GRAPH_HAS_CUDA
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu};
#else
        auto const deviceSpec = alpaka::onHost::DeviceSpec{alpaka::api::host, alpaka::deviceKind::cpu};
#endif
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        REQUIRE(selector.getDeviceCount() > 0u);
        return selector.makeDevice(0);
    }

    auto makeLaunchCfg([[maybe_unused]] std::uint32_t blocks, [[maybe_unused]] std::uint32_t threads)
    {
#if GRAPH_HAS_CUDA
        return alpaka::onHost::ThreadSpec{blocks, threads, alpaka::exec::gpuCuda};
#else
        // The serial CPU executor runs exactly one logical kernel invocation.
        return alpaka::onHost::FrameSpec{1u, 1u, alpaka::exec::cpuSerial};
#endif
    }

    template<typename T_Device>
    auto allocMappedValues(T_Device const& device, std::uint32_t count)
    {
        return alpaka::onHost::allocMapped<Value>(device, alpaka::Vec<std::uint32_t, 1u>{count});
    }

    Value hostLoadAcquire(Value* ptr)
    {
#if GRAPH_HAS_CUDA
        cuda::atomic_ref<Value, cuda::thread_scope_system> ref(*ptr);
#else
        std::atomic_ref<Value> ref(*ptr);
#endif
        return graph::host::hostLoadAcquire(ref);
    }

    void hostStoreRelease(Value* ptr, Value value)
    {
#if GRAPH_HAS_CUDA
        cuda::atomic_ref<Value, cuda::thread_scope_system> ref(*ptr);
#else
        std::atomic_ref<Value> ref(*ptr);
#endif
        graph::host::hostStoreRelease(ref, value);
    }

    void hostFetchAdd(Value* ptr, Value count)
    {
#if GRAPH_HAS_CUDA
        cuda::atomic_ref<Value, cuda::thread_scope_system> ref(*ptr);
        for(Value i = 0u; i < count; ++i)
            ref.fetch_add(Value{1}, cuda::std::memory_order_relaxed);
#else
        std::atomic_ref<Value> ref(*ptr);
        for(Value i = 0u; i < count; ++i)
            ref.fetch_add(Value{1}, std::memory_order_relaxed);
#endif
    }

    struct AddZeroKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto counter,
            alpaka::concepts::IMdSpan auto oldValues) const
        {
            oldValues[0] = graph::atomicAdd(acc, &counter[0], Value{0}, alpaka::onAcc::scope::system);
            oldValues[1] = graph::atomicAdd(acc, &counter[1], Value{7}, alpaka::onAcc::scope::device);
        }
    };

    struct DeviceAddKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto counter, Value iterations) const
        {
            auto* ptr = &counter[0];
            for(Value i = 0u; i < iterations; ++i)
                graph::atomicAdd(acc, ptr, Value{1}, alpaka::onAcc::scope::system);
        }
    };

    /** Consume descriptors published by the host.
     *
     * The consumer observes `head` with an acquire load, applies the system
     * acquire fence, verifies the payload, and acknowledges with a release store
     * to `tail`.
     */
    struct ConsumeDescriptorsKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto sequences,
            alpaka::concepts::IMdSpan auto payloads,
            alpaka::concepts::IMdSpan auto head,
            alpaka::concepts::IMdSpan auto tail,
            alpaka::concepts::IMdSpan auto errors,
            Value count,
            Value capacity) const
        {
            auto* headPtr = &head[0];
            auto* tailPtr = &tail[0];
            Value consumed = graph::systemLoadAcquire(acc, tailPtr);
            while(consumed < count)
            {
                Value const published = graph::systemLoadAcquire(acc, headPtr);
                if(published <= consumed)
                    continue;

                graph::systemFenceAcquire(acc);
                Value const slot = consumed % capacity;
                Value const sequence = sequences[slot];
                Value const payload = payloads[slot];
                if(sequence != consumed || payload != mix64(consumed))
                    errors[0] += Value{1};

                consumed += 1u;
                graph::systemStoreRelease(acc, tailPtr, consumed);
            }
        }
    };

    /** Publish completions from the accelerator to the host. */
    struct ProduceCompletionsKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto sequences,
            alpaka::concepts::IMdSpan auto payloads,
            alpaka::concepts::IMdSpan auto head,
            Value count,
            Value capacity) const
        {
            auto* headPtr = &head[0];
            Value const publishedStart = graph::systemLoadAcquire(acc, headPtr);
            for(Value i = 0u; i < count; ++i)
            {
                Value const sequence = publishedStart + i;
                Value const slot = sequence % capacity;
                sequences[slot] = sequence;
                payloads[slot] = mix64(sequence);
                graph::systemFenceRelease(acc);
                graph::systemStoreRelease(acc, headPtr, sequence + 1u);
            }
        }
    };
} // namespace

TEST_CASE("system_atomic: add-zero returns the old value and keeps memory unchanged", "[graph][system_atomic]")
{
    auto device = makeTestDevice();
    auto queue = device.makeQueue();

    auto counter = allocMappedValues(device, 2u);
    auto oldValues = allocMappedValues(device, 2u);
    auto* counterPtr = alpaka::onHost::data(counter);
    auto* oldPtr = alpaka::onHost::data(oldValues);
    counterPtr[0] = Value{42};
    counterPtr[1] = Value{42};
    oldPtr[0] = Value{0};
    oldPtr[1] = Value{0};

    queue.enqueue(makeLaunchCfg(1u, 1u), AddZeroKernel{}, counter, oldValues);
    alpaka::onHost::wait(queue);

    REQUIRE(oldPtr[0] == Value{42});
    // The add-zero operation must not change the counter.
    REQUIRE(counterPtr[0] == Value{42});
    // The device-scope overload forwards to upstream and observes the same value.
    REQUIRE(oldPtr[1] == Value{42});
    REQUIRE(counterPtr[1] == Value{49});
}

TEST_CASE("system_atomic: concurrent CPU and device adds on an eligible allocation", "[graph][system_atomic]")
{
    auto device = makeTestDevice();
    auto queue = device.makeQueue();
    auto const caps = graph::inquireSystemMemoryCapabilities(device);

    // A cross-system RMW needs an allocation the device can reach atomically
    // while the host is also issuing RMWs. Mapped pinned memory is eligible only
    // when the device reports host-native atomics. The A30 reports 0 and the
    // probe measured lost updates when host and device RMW the same mapped word
    // concurrently. Managed memory is the eligible allocation when
    // concurrentManagedAccess is set.
    REQUIRE((caps.hostNativeAtomicSupported || caps.concurrentManagedAccess));
    bool const useManaged = !caps.hostNativeAtomicSupported && caps.concurrentManagedAccess;

    auto mappedCounter = allocMappedValues(device, 1u);
    auto managedCounter = alpaka::onHost::allocUnified<Value>(device, alpaka::Vec<std::uint32_t, 1u>{1u});
    Value* counterPtr = useManaged ? alpaka::onHost::data(managedCounter) : alpaka::onHost::data(mappedCounter);
    counterPtr[0] = Value{0};

    constexpr Value deviceBlocks = 2u;
    constexpr Value deviceThreads = 32u;
    constexpr Value deviceIterations = 1000u;
    constexpr Value hostIterations = 5000u;
#if GRAPH_HAS_CUDA
    constexpr Value launchedThreads = deviceBlocks * deviceThreads;
#else
    constexpr Value launchedThreads = 1u;
#endif

    auto const launchCfg
        = makeLaunchCfg(static_cast<std::uint32_t>(deviceBlocks), static_cast<std::uint32_t>(deviceThreads));
    if(useManaged)
        queue.enqueue(launchCfg, DeviceAddKernel{}, managedCounter, deviceIterations);
    else
        queue.enqueue(launchCfg, DeviceAddKernel{}, mappedCounter, deviceIterations);
    // The queue is non-blocking on CUDA, so these host atomics overlap the
    // kernel. The sum is exact for any interleaving on an eligible allocation.
    hostFetchAdd(counterPtr, hostIterations);
    alpaka::onHost::wait(queue);

    REQUIRE(counterPtr[0] == launchedThreads * deviceIterations + hostIterations);
}

TEST_CASE("system_atomic: CPU backend and host std::atomic_ref path", "[graph][system_atomic]")
{
    auto hostDevice = alpaka::onHost::makeHostDevice();
    auto queue = hostDevice.makeQueue();

    auto counter = allocMappedValues(hostDevice, 1u);
    auto* counterPtr = alpaka::onHost::data(counter);
    counterPtr[0] = Value{0};

    constexpr Value iterations = 1000u;
    queue.enqueue(alpaka::onHost::FrameSpec{1u, 1u, alpaka::exec::cpuSerial}, DeviceAddKernel{}, counter, iterations);
    alpaka::onHost::wait(queue);
    REQUIRE(counterPtr[0] == iterations);

    std::atomic_ref<Value> ref(*counterPtr);
    constexpr std::uint32_t hostThreads = 4u;
    constexpr Value perThread = 2000u;
    std::vector<std::thread> workers;
    workers.reserve(hostThreads);
    for(std::uint32_t thread = 0u; thread < hostThreads; ++thread)
    {
        workers.emplace_back(
            [&ref]()
            {
                for(Value i = 0u; i < perThread; ++i)
                    ref.fetch_add(Value{1}, std::memory_order_relaxed);
            });
    }
    for(auto& worker : workers)
        worker.join();

    REQUIRE(counterPtr[0] == iterations + static_cast<Value>(hostThreads) * perThread);
}

TEST_CASE("system_atomic: host publishes descriptors the device consumes", "[graph][system_atomic]")
{
    auto device = makeTestDevice();
    auto queue = device.makeQueue();

    constexpr Value count = 256u;
    constexpr Value capacity = count;

    auto sequences = allocMappedValues(device, static_cast<std::uint32_t>(capacity));
    auto payloads = allocMappedValues(device, static_cast<std::uint32_t>(capacity));
    auto head = allocMappedValues(device, 1u);
    auto tail = allocMappedValues(device, 1u);
    auto errors = allocMappedValues(device, 1u);

    auto* sequencePtr = alpaka::onHost::data(sequences);
    auto* payloadPtr = alpaka::onHost::data(payloads);
    auto* headPtr = alpaka::onHost::data(head);
    auto* tailPtr = alpaka::onHost::data(tail);
    auto* errorPtr = alpaka::onHost::data(errors);
    headPtr[0] = Value{0};
    tailPtr[0] = Value{0};
    errorPtr[0] = Value{0};

    for(Value i = 0u; i < count; ++i)
    {
        sequencePtr[i] = i;
        payloadPtr[i] = mix64(i);
        graph::host::hostFenceRelease();
        hostStoreRelease(headPtr, i + 1u);
    }

    queue.enqueue(
        makeLaunchCfg(1u, 1u),
        ConsumeDescriptorsKernel{},
        sequences,
        payloads,
        head,
        tail,
        errors,
        count,
        capacity);
    alpaka::onHost::wait(queue);

    REQUIRE(errorPtr[0] == Value{0});
    REQUIRE(headPtr[0] == count);
    REQUIRE(tailPtr[0] == count);
    for(Value i = 0u; i < count; ++i)
    {
        REQUIRE(sequencePtr[i] == i);
        REQUIRE(payloadPtr[i] == mix64(i));
    }
}

TEST_CASE("system_atomic: device publishes completions the host consumes", "[graph][system_atomic]")
{
    auto device = makeTestDevice();
    auto queue = device.makeQueue();

    constexpr Value count = 256u;
    constexpr Value capacity = count;

    auto sequences = allocMappedValues(device, static_cast<std::uint32_t>(capacity));
    auto payloads = allocMappedValues(device, static_cast<std::uint32_t>(capacity));
    auto head = allocMappedValues(device, 1u);
    auto tail = allocMappedValues(device, 1u);

    auto* sequencePtr = alpaka::onHost::data(sequences);
    auto* payloadPtr = alpaka::onHost::data(payloads);
    auto* headPtr = alpaka::onHost::data(head);
    auto* tailPtr = alpaka::onHost::data(tail);
    headPtr[0] = Value{0};
    tailPtr[0] = Value{0};

    queue.enqueue(makeLaunchCfg(1u, 1u), ProduceCompletionsKernel{}, sequences, payloads, head, count, capacity);

    // The store is a system-scope release from the device. The host observes it
    // with an acquire load and then reads the payload.
    Value observed = hostLoadAcquire(headPtr);
    while(observed < count)
        observed = hostLoadAcquire(headPtr);
    graph::host::hostFenceAcquire();
    alpaka::onHost::wait(queue);

    REQUIRE(observed == count);
    for(Value i = 0u; i < count; ++i)
    {
        REQUIRE(sequencePtr[i] == i);
        REQUIRE(payloadPtr[i] == mix64(i));
    }
    hostStoreRelease(tailPtr, count);
    REQUIRE(tailPtr[0] == count);
}

TEST_CASE("system_atomic: transport selection rejects an unverified RMW", "[graph][system_atomic]")
{
    using graph::IngressTransport;
    using graph::SystemMemoryCapabilities;

    SystemMemoryCapabilities caps;
    caps.mappedPinnedAllocation = true;

    CHECK(graph::selectIngressTransport(caps) == IngressTransport::OrderedSystemAtomic);
    // The raw doorbell was requested but no live verification happened.
    CHECK(graph::selectIngressTransport(caps, true) == IngressTransport::Unsupported);

    caps.mappedSystemAtomicVerified = true;
    caps.deviceSystemScopeRmwNative = true;
    CHECK(graph::selectIngressTransport(caps) == IngressTransport::OrderedSystemAtomic);
    CHECK(graph::selectIngressTransport(caps, true) == IngressTransport::NativeSystemRmw);

    // Verified but the device RMW attribute is absent: do not silently degrade.
    caps.deviceSystemScopeRmwNative = false;
    CHECK(graph::selectIngressTransport(caps, true) == IngressTransport::Unsupported);

    // No mapped pinned allocation: unsupported even with every other flag set.
    caps.mappedPinnedAllocation = false;
    caps.deviceSystemScopeRmwNative = true;
    caps.mappedSystemAtomicVerified = true;
    CHECK(graph::selectIngressTransport(caps) == IngressTransport::Unsupported);
    CHECK(graph::selectIngressTransport(caps, true) == IngressTransport::Unsupported);
}

TEST_CASE("system_atomic: capability probe, live verification and selected transport", "[graph][system_atomic]")
{
    auto device = makeTestDevice();

    graph::SystemMemoryCapabilities const attributes = graph::inquireSystemMemoryCapabilities(device);
    graph::SystemAtomicVerification const verification = graph::verifyMappedSystemAtomics(device);
    graph::SystemMemoryCapabilities const effective = graph::inquireVerifiedSystemMemoryCapabilities(device);

    std::cout << "system_atomic capability record: device=\"" << device.getName() << "\""
              << " attribute{deviceSystemScopeRmwNative=" << attributes.deviceSystemScopeRmwNative
              << ",hostNativeAtomicSupported=" << attributes.hostNativeAtomicSupported
              << ",concurrentManagedAccess=" << attributes.concurrentManagedAccess
              << ",mappedPinnedAllocation=" << attributes.mappedPinnedAllocation
              << ",mappedSystemAtomicVerified=" << attributes.mappedSystemAtomicVerified << "}"
              << " verification{mappedPinnedAllocation=" << verification.mappedPinnedAllocation
              << ",expected=" << verification.expected << ",orderedObserved=" << verification.observedOrdered
              << ",nativeObserved=" << verification.observedNative
              << ",orderedVerified=" << verification.orderedAccessVerified
              << ",nativeVerified=" << verification.nativeSystemRmwVerified << "}"
              << " effective{deviceSystemScopeRmwNative=" << effective.deviceSystemScopeRmwNative
              << ",mappedSystemAtomicVerified=" << effective.mappedSystemAtomicVerified << "}"
              << " selected=" << graph::toString(graph::selectIngressTransport(effective))
              << " selectedNativeOptIn=" << graph::toString(graph::selectIngressTransport(effective, true))
              << std::endl;

    REQUIRE(attributes.mappedPinnedAllocation);
    REQUIRE(verification.mappedPinnedAllocation);
    REQUIRE(verification.orderedAccessVerified);
    REQUIRE(verification.observedOrdered == verification.expected);
    // The raw path is measured behaviour. Record the outcome instead of
    // requiring it on devices that do not implement it.
    CHECK(verification.observedNative == verification.expected);
    CHECK(verification.nativeSystemRmwVerified == (verification.observedNative == verification.expected));
    REQUIRE(graph::selectIngressTransport(effective) == graph::IngressTransport::OrderedSystemAtomic);
}
