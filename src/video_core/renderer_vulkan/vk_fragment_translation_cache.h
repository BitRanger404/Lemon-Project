// SPDX-FileCopyrightText: Copyright 2026 Lemon Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"

namespace Vulkan {

/// One question a translation asked its environment, and the answer it got.
struct EnvironmentQuery {
    enum class Kind : u8 {
        CbufValue,
        TextureType,
        TexturePixelFormat,
        TexturePixelFormatInteger,
        ViewportTransformState,
        TextureBoundBuffer,
        LocalMemorySize,
        SharedMemorySize,
        WorkgroupSize,
        HLEMacroState,
        ReplaceConstBuffer,
    };
    Kind kind{};
    u32 first{};
    u32 second{};
    u64 answer{};
};

/// Forwards everything to another environment and records every question asked but the
/// instruction reads, which the shader hash already covers.
class RecordingEnvironment final : public Shader::Environment {
public:
    explicit RecordingEnvironment(Shader::Environment& env_);

    u64 ReadInstruction(u32 address) override;
    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;
    Shader::TextureType ReadTextureType(u32 raw_handle) override;
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 raw_handle) override;
    bool IsTexturePixelFormatInteger(u32 raw_handle) override;
    u32 ReadViewportTransformState() override;
    u32 TextureBoundBuffer() const override;
    u32 LocalMemorySize() const override;
    u32 SharedMemorySize() const override;
    std::array<u32, 3> WorkgroupSize() const override;
    bool HasHLEMacroState() const override;
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;
    void Dump(u64 pipeline_hash, u64 shader_hash) override;

    [[nodiscard]] std::vector<EnvironmentQuery> TakeQueries() {
        return std::move(queries);
    }

private:
    void Record(EnvironmentQuery::Kind kind, u32 first, u32 second, u64 answer) const;

    Shader::Environment& env;
    mutable std::vector<EnvironmentQuery> queries;
};

/// Fragment shaders that were expensive to translate, kept for the session so the pipelines that
/// share one don't translate it again. Splatoon 3 has a fragment shader that takes seconds and
/// gigabytes to translate, used by dozens of pipelines: translating it for each one made the game
/// load for minutes and run out of memory when two pipeline workers translated it at once.
///
/// A translation is reused when the new pipeline's environment gives the same answers to every
/// question the translation asked, and the pipeline state that shapes the SPIR-V (runtime info,
/// bindings, rescaling settings) is the same.
class FragmentTranslationCache {
public:
    struct Result {
        std::vector<u32> code;
        Shader::Info info;
    };

    /// Held while a shader is looked up and, on a miss, translated: pipelines that share it wait
    /// for the first translation instead of running their own in parallel.
    class Slot {
    public:
        [[nodiscard]] std::optional<Result> Find(Shader::Environment& env,
                                                 const std::vector<u8>& variant_key) const;
        void Add(std::vector<EnvironmentQuery> queries, std::vector<u8> variant_key, Result result);

    private:
        friend class FragmentTranslationCache;

        struct Entry {
            std::vector<EnvironmentQuery> queries;
            std::vector<u8> variant_key;
            Result result;
        };
        std::mutex mutex;
        std::vector<Entry> entries;
    };

    /// Slot of a shader, locked.
    [[nodiscard]] std::pair<Slot*, std::unique_lock<std::mutex>> Acquire(u64 shader_hash);

    /// Bytes of everything outside the shader and its environment that changes the SPIR-V.
    [[nodiscard]] static std::vector<u8> MakeVariantKey(const Shader::RuntimeInfo& runtime_info,
                                                        const Shader::Backend::Bindings& bindings);

private:
    std::mutex map_mutex;
    std::unordered_map<u64, std::unique_ptr<Slot>> slots;
};

} // namespace Vulkan
