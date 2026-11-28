#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "squatris/core/pose/mapping.hpp"
#include "squatris/core/result.hpp"

namespace squatris::backends::rtmpose {

struct PersonBox {
  float left{};
  float top{};
  float right{};
  float bottom{};
};

struct PoseRegion {
  float center_x{};
  float center_y{};
  float width{};
  float height{};
};

[[nodiscard]] constexpr bool tensor_metadata_dimension_matches(
    std::int64_t actual, std::int64_t expected, bool allow_dynamic) noexcept {
  return expected == 0 || actual == expected || (allow_dynamic && actual == -1);
}

[[nodiscard]] std::array<float, 3>
normalize_pose_pixel(std::array<std::uint8_t, 3> bgr_pixel) noexcept;

[[nodiscard]] Result<PoseRegion> prepare_pose_region(PersonBox box, float padding = 1.25F);

[[nodiscard]] Result<std::vector<pose::ModelLandmark>>
decode_simcc(std::span<const float> simcc_x, std::span<const float> simcc_y,
             std::size_t keypoint_count, std::size_t x_bins, std::size_t y_bins, PoseRegion region,
             float split_ratio = 2.0F);

}
