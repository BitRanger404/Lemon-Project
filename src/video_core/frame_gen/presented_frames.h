// SPDX-FileCopyrightText: Copyright 2026 Lemon Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

namespace VideoCore::FrameGen {

/// Frames handed to the display, real and generated: with frame generation on, the rate the
/// player actually sees, which the game's own FPS doesn't count.
inline std::atomic<u64> presented_frames{0};

} // namespace VideoCore::FrameGen
