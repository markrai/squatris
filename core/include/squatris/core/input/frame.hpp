#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "squatris/core/pose/pose.hpp"

namespace squatris::input {

enum class PixelFormat { bgr8 };

struct Frame {
  std::uint64_t sequence{};
  pose::TimePoint captured_at{};
  std::chrono::microseconds acquisition_duration{};
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint32_t stride{};
  PixelFormat pixel_format{PixelFormat::bgr8};
  std::vector<std::uint8_t> pixels;
};

}
