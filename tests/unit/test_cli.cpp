// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_test_macros.hpp>
#include <particle_feedback/cli.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{
    using graph::particle_feedback::Backend;
    using graph::particle_feedback::ParseResult;

    auto parseArgs(std::vector<std::string_view> arguments) -> ParseResult
    {
        return graph::particle_feedback::parse(std::span<std::string_view const>{arguments});
    }

    auto contains(std::string const& text, std::string const& needle) -> bool
    {
        return text.find(needle) != std::string::npos;
    }
} // namespace

static_assert(std::is_trivially_copyable_v<graph::particle_feedback::Options>);
static_assert(std::is_standard_layout_v<graph::particle_feedback::Options>);

TEST_CASE("cli defaults match the documented values", "[graph][cli]")
{
    auto const result = parseArgs({});

    REQUIRE(result.ok);
    CHECK_FALSE(result.help);
    CHECK(result.error.empty());
    CHECK(result.options.backend == Backend::Cpu);
    CHECK(result.options.particles == graph::particle_feedback::defaultParticleCount);
    CHECK(result.options.particles == 4096u);
    CHECK(result.options.chunkSize == graph::particle_feedback::defaultChunkSize);
    CHECK(result.options.chunkSize == 128u);
    CHECK(result.options.seed == graph::particle_feedback::defaultSeed);
    CHECK(result.options.seed == 42u);
}

TEST_CASE("cli accepts every flag", "[graph][cli]")
{
    auto const result = parseArgs({"--backend", "cuda", "--particles", "123", "--chunk-size", "17", "--seed", "99"});

    REQUIRE(result.ok);
    CHECK(result.options.backend == Backend::Cuda);
    CHECK(result.options.particles == 123u);
    CHECK(result.options.chunkSize == 17u);
    CHECK(result.options.seed == 99u);
}

TEST_CASE("cli accepts the --flag=value form", "[graph][cli]")
{
    auto const result = parseArgs({"--backend=cuda", "--particles=1", "--chunk-size=2", "--seed=3"});

    REQUIRE(result.ok);
    CHECK(result.options.backend == Backend::Cuda);
    CHECK(result.options.particles == 1u);
    CHECK(result.options.chunkSize == 2u);
    CHECK(result.options.seed == 3u);
}

TEST_CASE("cli accepts N=0", "[graph][cli]")
{
    auto const result = parseArgs({"--particles", "0"});

    REQUIRE(result.ok);
    CHECK(result.options.particles == 0u);
}

TEST_CASE("cli accepts a zero seed", "[graph][cli]")
{
    auto const result = parseArgs({"--seed", "0"});

    REQUIRE(result.ok);
    CHECK(result.options.seed == 0u);
}

TEST_CASE("cli rejects negative values", "[graph][cli]")
{
    auto const particles = parseArgs({"--particles", "-1"});
    CHECK_FALSE(particles.ok);
    CHECK(contains(particles.error, "invalid value '-1'"));
    CHECK(contains(particles.error, "non-negative integer"));

    auto const chunk = parseArgs({"--chunk-size", "-5"});
    CHECK_FALSE(chunk.ok);
    CHECK(contains(chunk.error, "invalid value '-5'"));

    auto const seed = parseArgs({"--seed", "-2"});
    CHECK_FALSE(seed.ok);
    CHECK(contains(seed.error, "invalid value '-2'"));
}

TEST_CASE("cli rejects a zero chunk size", "[graph][cli]")
{
    auto const result = parseArgs({"--chunk-size", "0"});

    CHECK_FALSE(result.ok);
    CHECK(contains(result.error, "must be at least 1"));
}

TEST_CASE("cli rejects non-numeric values", "[graph][cli]")
{
    auto const particles = parseArgs({"--particles", "many"});
    CHECK_FALSE(particles.ok);
    CHECK(contains(particles.error, "invalid value 'many'"));

    auto const chunk = parseArgs({"--chunk-size", "12x"});
    CHECK_FALSE(chunk.ok);

    auto const seed = parseArgs({"--seed", "1.5"});
    CHECK_FALSE(seed.ok);
}

TEST_CASE("cli rejects an unknown backend", "[graph][cli]")
{
    auto const result = parseArgs({"--backend", "gpu"});

    CHECK_FALSE(result.ok);
    CHECK(contains(result.error, "unknown backend 'gpu'"));
    CHECK(contains(result.error, "'cpu' or 'cuda'"));
}

TEST_CASE("cli rejects unknown options, missing values, and stray arguments", "[graph][cli]")
{
    auto const unknown = parseArgs({"--bogus"});
    CHECK_FALSE(unknown.ok);
    CHECK(contains(unknown.error, "unknown option '--bogus'"));

    auto const missing = parseArgs({"--particles"});
    CHECK_FALSE(missing.ok);
    CHECK(contains(missing.error, "option '--particles' requires a value"));

    auto const stray = parseArgs({"hello"});
    CHECK_FALSE(stray.ok);
    CHECK(contains(stray.error, "unexpected argument 'hello'"));
}

TEST_CASE("cli --help reports help and success", "[graph][cli]")
{
    auto const longHelp = parseArgs({"--help"});
    CHECK(longHelp.ok);
    CHECK(longHelp.help);
    CHECK(longHelp.error.empty());

    auto const shortHelp = parseArgs({"-h"});
    CHECK(shortHelp.ok);
    CHECK(shortHelp.help);
}
