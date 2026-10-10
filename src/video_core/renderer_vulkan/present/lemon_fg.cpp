// SPDX-FileCopyrightText: Copyright 2026 Lemon Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <vector>

#include "video_core/host_shaders/lemon_fg_generate_comp_spv.h"
#include "video_core/host_shaders/lemon_fg_luma_comp_spv.h"
#include "video_core/host_shaders/lemon_fg_median_comp_spv.h"
#include "video_core/host_shaders/lemon_fg_project_comp_spv.h"
#include "video_core/host_shaders/lemon_fg_search_comp_spv.h"
#include "video_core/renderer_vulkan/present/lemon_fg.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

constexpr VkFormat WORK_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr u32 BLOCK = 8;
constexpr u32 SEARCH_REDUCTION = 4; // motion is searched at a quarter of the output resolution

// Penalties of the motion search, in SAD units (0-255 scale) per pixel of vector length, per
// block row: chosen on captured gameplay (scratchpad harness), see lemon_fg_search.comp.
constexpr f32 ZERO_BIAS = 0.2f;
constexpr f32 SMOOTH_BIAS = 1.0f;

enum SearchMode : u32 {
    MODE_EXHAUSTIVE = 0,
    MODE_FROM_COARSER = 1,
    MODE_REFINE = 2,
};

[[nodiscard]] u32 DivCeil(u32 value, u32 divisor) {
    return (value + divisor - 1) / divisor;
}

[[nodiscard]] VkExtent2D GridOf(VkExtent2D size) {
    return {DivCeil(size.width, BLOCK), DivCeil(size.height, BLOCK)};
}

[[nodiscard]] u32 Bits(f32 value) {
    return std::bit_cast<u32>(value);
}

} // Anonymous namespace

LemonFgChain::LemonFgChain(const Device& device_, MemoryAllocator& memory_allocator,
                           VkExtent2D extent_, VkFormat format)
    : extent{extent_}, device{&device_} {
    level_size[0] = {std::max(1u, DivCeil(extent.width, SEARCH_REDUCTION)),
                     std::max(1u, DivCeil(extent.height, SEARCH_REDUCTION))};
    for (size_t level = 1; level < LEVELS; ++level) {
        level_size[level] = {std::max(1u, DivCeil(level_size[level - 1].width, 2)),
                             std::max(1u, DivCeil(level_size[level - 1].height, 2))};
    }
    for (size_t level = 0; level < LEVELS; ++level) {
        grid[level] = GridOf(level_size[level]);
    }

    descriptor_pool = CreateWrappedDescriptorPool(
        *device, 1024, 256,
        {VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    sampler = CreateLsfgSampler(*device, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                VK_COMPARE_OP_NEVER, false);

    constexpr auto SAMPLER = VK_DESCRIPTOR_TYPE_SAMPLER;
    constexpr auto SAMPLED = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    constexpr auto STORAGE = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    luma_pass = CreatePass(*device, LEMON_FG_LUMA_COMP_SPV, {SAMPLER, SAMPLED, STORAGE},
                           sizeof(u32) * 5);
    search_pass = CreatePass(*device, LEMON_FG_SEARCH_COMP_SPV,
                             {SAMPLER, SAMPLED, SAMPLED, SAMPLED, SAMPLED, STORAGE},
                             sizeof(u32) * 13);
    median_pass = CreatePass(*device, LEMON_FG_MEDIAN_COMP_SPV, {SAMPLER, SAMPLED, STORAGE},
                             sizeof(u32) * 2);
    project_pass = CreatePass(*device, LEMON_FG_PROJECT_COMP_SPV,
                              {SAMPLER, SAMPLED, SAMPLED, SAMPLED, SAMPLED, STORAGE},
                              sizeof(u32) * 5);
    generate_pass = CreatePass(*device, LEMON_FG_GENERATE_COMP_SPV,
                               {SAMPLER, SAMPLED, SAMPLED, SAMPLED, STORAGE}, sizeof(u32) * 7);

    for (auto& frame : frames) {
        frame = LsfgImage(*device, memory_allocator, extent, format);
    }
    for (auto& slot : pyramid) {
        for (size_t level = 0; level < LEVELS; ++level) {
            slot[level] = LsfgImage(*device, memory_allocator, level_size[level], WORK_FORMAT);
        }
    }
    for (auto& direction : fields) {
        for (size_t level = 0; level < LEVELS; ++level) {
            for (auto& image : direction[level]) {
                image = LsfgImage(*device, memory_allocator, grid[level], WORK_FORMAT);
            }
        }
    }
    for (auto& direction : flows) {
        for (auto& image : direction) {
            image = LsfgImage(*device, memory_allocator, grid[0], WORK_FORMAT);
        }
    }
    offsets = LsfgImage(*device, memory_allocator, grid[0], WORK_FORMAT);
    offsets_filtered = LsfgImage(*device, memory_allocator, grid[0], WORK_FORMAT);

    for (size_t slot = 0; slot < 2; ++slot) {
        luma_sets[slot] = AllocateSet(luma_pass);
        LsfgDescriptorWriter(luma_sets[slot])
            .AddSampler(*sampler)
            .AddSampledImage(frames[slot])
            .AddStorageImage(pyramid[slot][0])
            .Build(*device);
        for (size_t level = 1; level < LEVELS; ++level) {
            VkDescriptorSet& set = down_sets[slot][level - 1];
            set = AllocateSet(luma_pass);
            LsfgDescriptorWriter(set)
                .AddSampler(*sampler)
                .AddSampledImage(pyramid[slot][level - 1])
                .AddStorageImage(pyramid[slot][level])
                .Build(*device);
        }

        // The pair ending at `slot`: A is the other slot's frame, C this one's.
        const auto& luma_a = pyramid[1 - slot];
        const auto& luma_c = pyramid[slot];
        for (size_t direction = 0; direction < 2; ++direction) {
            const f32 ka = direction == 0 ? 0.0f : 1.0f;
            const f32 kc = direction == 0 ? 1.0f : 0.0f;
            auto& steps = search_steps[slot][direction];
            size_t next = 0;
            const auto add_step = [&](size_t level, u32 mode, const LsfgImage& in,
                                      VkExtent2D in_grid, LsfgImage& out) {
                SearchStep& step = steps[next++];
                step.set = AllocateSet(search_pass);
                step.grid = grid[level];
                LsfgDescriptorWriter(step.set)
                    .AddSampler(*sampler)
                    .AddSampledImage(luma_a[level])
                    .AddSampledImage(luma_c[level])
                    .AddSampledImage(in)
                    .AddSampledImage(flows[direction][1 - slot])
                    .AddStorageImage(out)
                    .Build(*device);
                step.push = {
                    level_size[level].width, level_size[level].height,
                    grid[level].width,       grid[level].height,
                    in_grid.width,           in_grid.height,
                    Bits(ka),                Bits(kc),
                    mode,                    0, // use_temporal, set when recording
                    Bits(1.0f / static_cast<f32>(1u << level)),
                    Bits(ZERO_BIAS),         Bits(SMOOTH_BIAS),
                };
            };
            auto& coarsest = fields[direction][LEVELS - 1];
            add_step(LEVELS - 1, MODE_EXHAUSTIVE, coarsest[1], grid[LEVELS - 1], coarsest[0]);
            add_step(LEVELS - 1, MODE_REFINE, coarsest[0], grid[LEVELS - 1], coarsest[1]);
            add_step(LEVELS - 1, MODE_REFINE, coarsest[1], grid[LEVELS - 1], coarsest[0]);
            for (size_t level = LEVELS - 1; level-- > 0;) {
                const LsfgImage& coarser =
                    level + 1 == LEVELS - 1 ? coarsest[0] : fields[direction][level + 1][1];
                auto& current = fields[direction][level];
                add_step(level, MODE_FROM_COARSER, coarser, grid[level + 1], current[0]);
                add_step(level, MODE_REFINE, current[0], grid[level], current[1]);
            }

            VkDescriptorSet& median = flow_median_sets[slot][direction];
            median = AllocateSet(median_pass);
            LsfgDescriptorWriter(median)
                .AddSampler(*sampler)
                .AddSampledImage(fields[direction][0][1])
                .AddStorageImage(flows[direction][slot])
                .Build(*device);
        }

        project_sets[slot] = AllocateSet(project_pass);
        LsfgDescriptorWriter(project_sets[slot])
            .AddSampler(*sampler)
            .AddSampledImage(luma_a[0])
            .AddSampledImage(luma_c[0])
            .AddSampledImage(flows[0][slot])
            .AddSampledImage(flows[1][slot])
            .AddStorageImage(offsets)
            .Build(*device);
        for (auto& set : generate_sets[slot]) {
            set = AllocateSet(generate_pass);
        }
    }
    offsets_median_set = AllocateSet(median_pass);
    LsfgDescriptorWriter(offsets_median_set)
        .AddSampler(*sampler)
        .AddSampledImage(offsets)
        .AddStorageImage(offsets_filtered)
        .Build(*device);
}

bool LemonFgChain::IsFormatSupported(VkFormat format) {
    // The generate shader writes through an rgba8 storage image.
    return format == VK_FORMAT_R8G8B8A8_UNORM;
}

LemonFgChain::Pass LemonFgChain::CreatePass(const Device& device_, std::span<const u32> code,
                                            std::initializer_list<VkDescriptorType> bindings,
                                            u32 push_size) {
    Pass pass;
    pass.push_size = push_size;
    pass.module = CreateWrappedShaderModule(device_, code);
    const std::vector<VkDescriptorType> types(bindings);
    pass.set_layout = CreateWrappedDescriptorSetLayout(
        device_, std::span<const VkDescriptorType>{types}, VK_SHADER_STAGE_COMPUTE_BIT);
    const VkDescriptorSetLayout set_layout = *pass.set_layout;
    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_size,
    };
    pass.pipeline_layout = device_.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
    pass.pipeline = CreateWrappedComputePipeline(device_, pass.pipeline_layout, *pass.module);
    return pass;
}

VkDescriptorSet LemonFgChain::AllocateSet(const Pass& pass) {
    VkDescriptorSetLayout layout = *pass.set_layout;
    owned_sets.push_back(CreateWrappedDescriptorSets(descriptor_pool, layout));
    return owned_sets.back()[0];
}

void LemonFgChain::Run(vk::CommandBuffer cmdbuf, const Pass& pass, VkDescriptorSet set,
                       const void* push, u32 groups_x, u32 groups_y) {
    cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pass.pipeline);
    cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pass.pipeline_layout, 0, set, {});
    cmdbuf.PushConstants(*pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pass.push_size,
                         push);
    cmdbuf.Dispatch(groups_x, groups_y, 1);
    static constexpr VkMemoryBarrier WRITE_TO_READ{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, WRITE_TO_READ);
}

void LemonFgChain::InitializeImages(vk::CommandBuffer cmdbuf) {
    LsfgBarriers barriers(cmdbuf);
    for (auto& slot : pyramid) {
        barriers.ReadToWriteAll(slot);
    }
    for (auto& direction : fields) {
        for (auto& level : direction) {
            barriers.ReadToWriteAll(level);
        }
    }
    for (auto& direction : flows) {
        barriers.ReadToWriteAll(direction);
    }
    barriers.ReadToWrite(offsets).ReadToWrite(offsets_filtered).Build();
    initialized = true;
}

void LemonFgChain::BuildPyramid(vk::CommandBuffer cmdbuf, size_t slot) {
    const std::array<u32, 5> luma_push{extent.width, extent.height, level_size[0].width,
                                       level_size[0].height, 0};
    Run(cmdbuf, luma_pass, luma_sets[slot], luma_push.data(), DivCeil(level_size[0].width, 8),
        DivCeil(level_size[0].height, 8));
    for (size_t level = 1; level < LEVELS; ++level) {
        const std::array<u32, 5> down_push{level_size[level - 1].width,
                                           level_size[level - 1].height, level_size[level].width,
                                           level_size[level].height, 1};
        Run(cmdbuf, luma_pass, down_sets[slot][level - 1], down_push.data(),
            DivCeil(level_size[level].width, 8), DivCeil(level_size[level].height, 8));
    }
}

void LemonFgChain::DispatchShared(vk::CommandBuffer cmdbuf, u64 frame_count) {
    if (!initialized) {
        InitializeImages(cmdbuf);
    }
    const size_t slot = frame_count % 2;
    if (last_pyramid + 1 != frame_count) {
        // The previous frame was only copied, never analysed (first pair after a rebuild).
        BuildPyramid(cmdbuf, 1 - slot);
    }
    BuildPyramid(cmdbuf, slot);
    last_pyramid = frame_count;

    const u32 use_temporal = last_flow + 1 == frame_count ? 1 : 0;
    for (size_t direction = 0; direction < 2; ++direction) {
        for (SearchStep& step : search_steps[slot][direction]) {
            step.push[9] = use_temporal;
            // One workgroup per block.
            Run(cmdbuf, search_pass, step.set, step.push.data(), step.grid.width,
                step.grid.height);
        }
        const std::array<u32, 2> median_push{grid[0].width, grid[0].height};
        Run(cmdbuf, median_pass, flow_median_sets[slot][direction], median_push.data(),
            DivCeil(grid[0].width, 8), DivCeil(grid[0].height, 8));
    }
    last_flow = frame_count;
}

void LemonFgChain::SetTarget(const Device& device_, size_t, size_t, u32 target,
                             VkImageView view) {
    if (target_views[target] == view) {
        return;
    }
    target_views[target] = view;
    for (size_t slot = 0; slot < 2; ++slot) {
        LsfgDescriptorWriter(generate_sets[slot][target])
            .AddSampler(*sampler)
            .AddSampledImage(frames[1 - slot])
            .AddSampledImage(frames[slot])
            .AddSampledImage(offsets_filtered)
            .AddStorageView(view)
            .Build(device_);
    }
}

void LemonFgChain::DispatchGeneration(vk::CommandBuffer cmdbuf, u64 frame_count,
                                      size_t generation_count, size_t generation, u32 target,
                                      VkImage image, VkExtent2D target_extent) {
    const size_t slot = frame_count % 2;
    const f32 t = LsfgTimestamp(generation, generation_count);

    const std::array<u32, 5> project_push{level_size[0].width, level_size[0].height,
                                          grid[0].width, grid[0].height, Bits(t)};
    Run(cmdbuf, project_pass, project_sets[slot], project_push.data(), grid[0].width,
        grid[0].height);
    const std::array<u32, 2> median_push{grid[0].width, grid[0].height};
    Run(cmdbuf, median_pass, offsets_median_set, median_push.data(), DivCeil(grid[0].width, 8),
        DivCeil(grid[0].height, 8));

    LsfgBarriers(cmdbuf).DiscardToWrite(image).Build();
    const std::array<u32, 7> generate_push{target_extent.width, target_extent.height,
                                           level_size[0].width, level_size[0].height,
                                           grid[0].width,       grid[0].height,
                                           Bits(t)};
    Run(cmdbuf, generate_pass, generate_sets[slot][target], generate_push.data(),
        DivCeil(target_extent.width, 16), DivCeil(target_extent.height, 8));

    const VkImageMemoryBarrier after{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, after);
}

} // namespace Vulkan
