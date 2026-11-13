// SPDX-FileCopyrightText: 2026 René Widera and contributors
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

/** Command line interface of the particle-feedback example (header only).
 *
 * The later `main.cpp` (a different milestone) only has to:
 *
 *     auto const result = graph::particle_feedback::parse(argc, argv);
 *     if(result.help)
 *         std::cout << graph::particle_feedback::usageText;
 *     else if(!result.ok)
 *         std::cerr << result.error << '\n';
 *     else
 *         run(result.options);
 *
 * `--help` reports `ok == true, help == true` and exits successfully. A parse
 * error reports `ok == false` and a message suitable for stderr; the caller
 * exits with a non-zero status.
 *
 * # Defaults
 *
 * | option         | default | meaning                                         |
 * | -------------- | ------- | ----------------------------------------------- |
 * | `--backend`    | `cpu`   | execution backend, `cpu` or `cuda`              |
 * | `--particles`  | 4096    | number of source particles `N`, `N >= 0`        |
 * | `--chunk-size` | 128     | maximum particles per chunk `Y`, `Y >= 1`       |
 * | `--seed`       | 42      | deterministic random-stream seed, `S >= 0`      |
 *
 * The defaults describe the design's `Y=128`; `N=4096` and seed `42` keep a
 * default run short (multiple chunks and multiple worker rounds) while the
 * acceptance runs pass an explicit `N >= 65536`.
 *
 * `N=0` is accepted and produces an empty histogram. Negative or non-numeric
 * values, a zero chunk size, an unknown backend, an unknown option, and a
 * missing option value are rejected with a message.
 *
 * Both `--option value` and `--option=value` are accepted.
 */
namespace graph::particle_feedback
{
    /** Execution backend selected on the command line. */
    enum class Backend : std::uint8_t
    {
        Cpu = 0u,
        Cuda = 1u
    };

    /** Default number of source particles. */
    inline constexpr std::uint64_t defaultParticleCount = 4096u;
    /** Default maximum number of particles per chunk (`Y` in the design). */
    inline constexpr std::uint32_t defaultChunkSize = 128u;
    /** Default deterministic random-stream seed. */
    inline constexpr std::uint64_t defaultSeed = 42u;

    /** Plain POD result of command line parsing. */
    struct Options
    {
        Backend backend{Backend::Cpu};
        std::uint64_t particles{defaultParticleCount};
        std::uint32_t chunkSize{defaultChunkSize};
        std::uint64_t seed{defaultSeed};
    };

    static_assert(std::is_trivially_copyable_v<Options>);
    static_assert(std::is_standard_layout_v<Options>);

    /** Canonical backend spelling accepted by `--backend`. */
    inline constexpr std::string_view backendName(Backend backend) noexcept
    {
        return backend == Backend::Cuda ? std::string_view{"cuda"} : std::string_view{"cpu"};
    }

    /** Help text printed for `--help`. */
    inline constexpr std::string_view usageText
        = "Usage: graph_particle_feedback [options]\n"
          "\n"
          "Options:\n"
          "  --backend cpu|cuda   Execution backend (default: cpu)\n"
          "  --particles N        Number of source particles, N >= 0 (default: 4096)\n"
          "  --chunk-size Y       Maximum particles per chunk, Y >= 1 (default: 128)\n"
          "  --seed S             Deterministic random-stream seed, S >= 0 (default: 42)\n"
          "  -h, --help           Print this help and exit\n";

    /** Parsing outcome. */
    struct ParseResult
    {
        bool ok{false};
        bool help{false};
        std::string error{};
        Options options{};

        explicit operator bool() const noexcept
        {
            return ok;
        }
    };

    namespace detail
    {
        inline ParseResult makeError(std::string message)
        {
            ParseResult result{};
            result.ok = false;
            result.error = std::move(message);
            return result;
        }

        /** Strict non-negative decimal parse. Rejects sign characters except a leading '+'. */
        inline bool parseUnsignedDecimal(std::string_view text, std::uint64_t& value) noexcept
        {
            if(text.empty())
                return false;
            if(text.front() == '-')
                return false;
            if(text.front() == '+')
                text.remove_prefix(1u);
            if(text.empty())
                return false;

            std::uint64_t parsed = 0u;
            auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
            if(error != std::errc{} || end != text.data() + text.size())
                return false;
            value = parsed;
            return true;
        }
    } // namespace detail

    /** Parse command line arguments (program name already removed).
     *
     * @param arguments options without `argv[0]`.
     */
    inline ParseResult parse(std::span<std::string_view const> arguments)
    {
        ParseResult result{};
        result.ok = true;

        for(std::size_t index = 0u; index < arguments.size(); ++index)
        {
            auto const argument = arguments[index];
            if(argument == "--help" || argument == "-h")
            {
                result.help = true;
                return result;
            }

            auto const equals = argument.find('=');
            auto const hasInlineValue = equals != std::string_view::npos;
            auto const name = argument.substr(0u, equals);
            auto const inlineValue = hasInlineValue ? argument.substr(equals + 1u) : std::string_view{};

            auto takeValue = [&](std::string_view& value) -> bool
            {
                if(hasInlineValue)
                {
                    value = inlineValue;
                    return true;
                }
                if(index + 1u >= arguments.size())
                    return false;
                value = arguments[++index];
                return true;
            };

            auto const valueError
                = [&]() { return detail::makeError("option '" + std::string{name} + "' requires a value"); };

            if(name == "--backend")
            {
                auto value = std::string_view{};
                if(!takeValue(value))
                    return valueError();
                if(value == "cpu")
                    result.options.backend = Backend::Cpu;
                else if(value == "cuda")
                    result.options.backend = Backend::Cuda;
                else
                    return detail::makeError("unknown backend '" + std::string{value} + "': expected 'cpu' or 'cuda'");
            }
            else if(name == "--particles")
            {
                auto value = std::string_view{};
                if(!takeValue(value))
                    return valueError();
                std::uint64_t parsed = 0u;
                if(!detail::parseUnsignedDecimal(value, parsed))
                    return detail::makeError(
                        "invalid value '" + std::string{value} + "' for --particles: expected a non-negative integer");
                result.options.particles = parsed;
            }
            else if(name == "--chunk-size")
            {
                auto value = std::string_view{};
                if(!takeValue(value))
                    return valueError();
                std::uint64_t parsed = 0u;
                if(!detail::parseUnsignedDecimal(value, parsed))
                    return detail::makeError(
                        "invalid value '" + std::string{value} + "' for --chunk-size: expected a positive integer");
                if(parsed == 0u)
                    return detail::makeError("invalid value '0' for --chunk-size: must be at least 1");
                if(parsed > std::numeric_limits<std::uint32_t>::max())
                    return detail::makeError(
                        "invalid value '" + std::string{value} + "' for --chunk-size: value is too large");
                result.options.chunkSize = static_cast<std::uint32_t>(parsed);
            }
            else if(name == "--seed")
            {
                auto value = std::string_view{};
                if(!takeValue(value))
                    return valueError();
                std::uint64_t parsed = 0u;
                if(!detail::parseUnsignedDecimal(value, parsed))
                    return detail::makeError(
                        "invalid value '" + std::string{value} + "' for --seed: expected a non-negative integer");
                result.options.seed = parsed;
            }
            else if(name.empty() || name.front() != '-')
            {
                return detail::makeError("unexpected argument '" + std::string{argument} + "'");
            }
            else
            {
                return detail::makeError("unknown option '" + std::string{name} + "' (try --help)");
            }
        }

        return result;
    }

    /** Parse `argc`/`argv`, skipping the program name. */
    inline ParseResult parse(int argc, char const* const* argv)
    {
        std::vector<std::string_view> arguments;
        arguments.reserve(argc > 0 ? static_cast<std::size_t>(argc - 1) : 0u);
        for(int index = 1; index < argc; ++index)
            arguments.emplace_back(argv[index]);
        return parse(std::span<std::string_view const>{arguments.data(), arguments.size()});
    }
} // namespace graph::particle_feedback
