// SPDX-FileCopyrightText: Copyright 2026 Lemon Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <cstring>

#include "common/settings.h"
#include "video_core/renderer_vulkan/vk_fragment_translation_cache.h"

namespace Vulkan {

namespace {

using Kind = EnvironmentQuery::Kind;

[[nodiscard]] u64 Answer(Shader::Environment& env, const EnvironmentQuery& query) {
    switch (query.kind) {
    case Kind::CbufValue:
        return env.ReadCbufValue(query.first, query.second);
    case Kind::TextureType:
        return static_cast<u64>(env.ReadTextureType(query.first));
    case Kind::TexturePixelFormat:
        return static_cast<u64>(env.ReadTexturePixelFormat(query.first));
    case Kind::TexturePixelFormatInteger:
        return env.IsTexturePixelFormatInteger(query.first) ? 1 : 0;
    case Kind::ViewportTransformState:
        return env.ReadViewportTransformState();
    case Kind::TextureBoundBuffer:
        return env.TextureBoundBuffer();
    case Kind::LocalMemorySize:
        return env.LocalMemorySize();
    case Kind::SharedMemorySize:
        return env.SharedMemorySize();
    case Kind::WorkgroupSize: {
        const std::array<u32, 3> size = env.WorkgroupSize();
        return size[query.first];
    }
    case Kind::HLEMacroState:
        return env.HasHLEMacroState() ? 1 : 0;
    case Kind::ReplaceConstBuffer: {
        const std::optional<Shader::ReplaceConstant> replace =
            env.GetReplaceConstBuffer(query.first, query.second);
        return replace ? static_cast<u64>(*replace) + 1 : 0;
    }
    }
    return ~0ULL;
}

template <typename T>
void Append(std::vector<u8>& bytes, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(T));
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

} // Anonymous namespace

RecordingEnvironment::RecordingEnvironment(Shader::Environment& env_) : env{env_} {
    sph = env.SPH();
    gp_passthrough_mask = env.GpPassthroughMask();
    stage = env.ShaderStage();
    start_address = env.StartAddress();
    is_proprietary_driver = env.IsProprietaryDriver();
}

void RecordingEnvironment::Record(Kind kind, u32 first, u32 second, u64 answer) const {
    queries.push_back({.kind = kind, .first = first, .second = second, .answer = answer});
}

u64 RecordingEnvironment::ReadInstruction(u32 address) {
    return env.ReadInstruction(address);
}

u32 RecordingEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const u32 value = env.ReadCbufValue(cbuf_index, cbuf_offset);
    Record(Kind::CbufValue, cbuf_index, cbuf_offset, value);
    return value;
}

Shader::TextureType RecordingEnvironment::ReadTextureType(u32 raw_handle) {
    const Shader::TextureType value = env.ReadTextureType(raw_handle);
    Record(Kind::TextureType, raw_handle, 0, static_cast<u64>(value));
    return value;
}

Shader::TexturePixelFormat RecordingEnvironment::ReadTexturePixelFormat(u32 raw_handle) {
    const Shader::TexturePixelFormat value = env.ReadTexturePixelFormat(raw_handle);
    Record(Kind::TexturePixelFormat, raw_handle, 0, static_cast<u64>(value));
    return value;
}

bool RecordingEnvironment::IsTexturePixelFormatInteger(u32 raw_handle) {
    const bool value = env.IsTexturePixelFormatInteger(raw_handle);
    Record(Kind::TexturePixelFormatInteger, raw_handle, 0, value ? 1 : 0);
    return value;
}

u32 RecordingEnvironment::ReadViewportTransformState() {
    const u32 value = env.ReadViewportTransformState();
    Record(Kind::ViewportTransformState, 0, 0, value);
    return value;
}

u32 RecordingEnvironment::TextureBoundBuffer() const {
    const u32 value = env.TextureBoundBuffer();
    Record(Kind::TextureBoundBuffer, 0, 0, value);
    return value;
}

u32 RecordingEnvironment::LocalMemorySize() const {
    const u32 value = env.LocalMemorySize();
    Record(Kind::LocalMemorySize, 0, 0, value);
    return value;
}

u32 RecordingEnvironment::SharedMemorySize() const {
    const u32 value = env.SharedMemorySize();
    Record(Kind::SharedMemorySize, 0, 0, value);
    return value;
}

std::array<u32, 3> RecordingEnvironment::WorkgroupSize() const {
    const std::array<u32, 3> value = env.WorkgroupSize();
    for (u32 i = 0; i < 3; ++i) {
        Record(Kind::WorkgroupSize, i, 0, value[i]);
    }
    return value;
}

bool RecordingEnvironment::HasHLEMacroState() const {
    const bool value = env.HasHLEMacroState();
    Record(Kind::HLEMacroState, 0, 0, value ? 1 : 0);
    return value;
}

std::optional<Shader::ReplaceConstant> RecordingEnvironment::GetReplaceConstBuffer(u32 bank,
                                                                                  u32 offset) {
    const std::optional<Shader::ReplaceConstant> value = env.GetReplaceConstBuffer(bank, offset);
    Record(Kind::ReplaceConstBuffer, bank, offset, value ? static_cast<u64>(*value) + 1 : 0);
    return value;
}

void RecordingEnvironment::Dump(u64 pipeline_hash, u64 shader_hash) {
    env.Dump(pipeline_hash, shader_hash);
}

std::optional<FragmentTranslationCache::Result> FragmentTranslationCache::Slot::Find(
    Shader::Environment& env, const std::vector<u8>& variant_key) const {
    for (const Entry& entry : entries) {
        if (entry.variant_key != variant_key) {
            continue;
        }
        // Ask the new environment the same questions. This also leaves the answers recorded in
        // it, as the translation would have, so the pipeline still serializes to the disk cache
        // with everything it needs to load next time.
        bool same = true;
        try {
            for (const EnvironmentQuery& query : entry.queries) {
                if (Answer(env, query) != query.answer) {
                    same = false;
                    break;
                }
            }
        } catch (...) {
            // A disk cache environment without that answer recorded: not the same translation.
            same = false;
        }
        if (same) {
            return entry.result;
        }
    }
    return std::nullopt;
}

void FragmentTranslationCache::Slot::Add(std::vector<EnvironmentQuery> queries,
                                         std::vector<u8> variant_key, Result result) {
    entries.push_back({.queries = std::move(queries),
                       .variant_key = std::move(variant_key),
                       .result = std::move(result)});
}

std::pair<FragmentTranslationCache::Slot*, std::unique_lock<std::mutex>>
FragmentTranslationCache::Acquire(u64 shader_hash) {
    Slot* slot;
    {
        std::scoped_lock lock{map_mutex};
        std::unique_ptr<Slot>& entry = slots[shader_hash];
        if (!entry) {
            entry = std::make_unique<Slot>();
        }
        slot = entry.get();
    }
    return {slot, std::unique_lock{slot->mutex}};
}

std::vector<u8> FragmentTranslationCache::MakeVariantKey(const Shader::RuntimeInfo& info,
                                                         const Shader::Backend::Bindings& bindings) {
    std::vector<u8> bytes;
    bytes.reserve(2048);
    Append(bytes, info.generic_input_types);
    for (size_t i = 0; i < info.previous_stage_stores.mask.size(); i += 64) {
        u64 word = 0;
        for (size_t bit = 0; bit < 64 && i + bit < info.previous_stage_stores.mask.size(); ++bit) {
            word |= static_cast<u64>(info.previous_stage_stores.mask[i + bit]) << bit;
        }
        Append(bytes, word);
    }
    Append(bytes, info.previous_stage_legacy_stores_mapping.size());
    for (const auto& [from, to] : info.previous_stage_legacy_stores_mapping) {
        Append(bytes, from);
        Append(bytes, to);
    }
    Append(bytes, info.convert_depth_mode);
    Append(bytes, info.force_early_z);
    Append(bytes, info.tess_primitive);
    Append(bytes, info.tess_spacing);
    Append(bytes, info.tess_clockwise);
    Append(bytes, info.input_topology);
    Append(bytes, info.fixed_state_point_size.has_value());
    Append(bytes, info.fixed_state_point_size.value_or(0.0f));
    Append(bytes, info.alpha_test_func.has_value());
    Append(bytes, info.alpha_test_func.value_or(Shader::CompareFunction{}));
    Append(bytes, info.alpha_test_reference);
    Append(bytes, info.y_negate);
    Append(bytes, info.glasm_use_storage_buffers);
    Append(bytes, info.xfb_count);
    for (u32 i = 0; i < info.xfb_count && i < info.xfb_varyings.size(); ++i) {
        const Shader::TransformFeedbackVarying& varying = info.xfb_varyings[i];
        Append(bytes, varying.buffer);
        Append(bytes, varying.stream);
        Append(bytes, varying.stride);
        Append(bytes, varying.offset);
        Append(bytes, varying.components);
    }
    Append(bytes, info.color_output_types);
    Append(bytes, info.dual_source_blend);

    Append(bytes, bindings.unified);
    Append(bytes, bindings.uniform_buffer);
    Append(bytes, bindings.storage_buffer);
    Append(bytes, bindings.texture);
    Append(bytes, bindings.image);
    Append(bytes, bindings.texture_scaling_index);
    Append(bytes, bindings.image_scaling_index);

    // Settings the translation and the SPIR-V emitter read.
    const auto& resolution = Settings::values.resolution_info;
    Append(bytes, resolution.active);
    Append(bytes, resolution.up_scale);
    Append(bytes, resolution.down_shift);
    Append(bytes, resolution.up_factor);
    Append(bytes, resolution.down_factor);
    Append(bytes, Settings::values.rescale_hack.GetValue());
    Append(bytes, Settings::values.fix_bloom_effects.GetValue());
    Append(bytes, Settings::values.disable_shader_loop_safety_checks.GetValue());
    Append(bytes, Settings::values.renderer_debug.GetValue());
    return bytes;
}

} // namespace Vulkan
