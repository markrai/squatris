#include "squatris/backends/rtmpose/model_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>

namespace squatris::backends::rtmpose {

std::array<float, 3> normalize_pose_pixel(std::array<std::uint8_t, 3> bgr_pixel) noexcept {
  constexpr std::array mean{123.675F, 116.28F, 103.53F};
  constexpr std::array standard_deviation{58.395F, 57.12F, 57.375F};
  return {(static_cast<float>(bgr_pixel[2]) - mean[0]) / standard_deviation[0],
          (static_cast<float>(bgr_pixel[1]) - mean[1]) / standard_deviation[1],
          (static_cast<float>(bgr_pixel[0]) - mean[2]) / standard_deviation[2]};
}

Result<PoseRegion> prepare_pose_region(PersonBox box, float padding) {
  if (!std::isfinite(box.left) || !std::isfinite(box.top) || !std::isfinite(box.right) ||
      !std::isfinite(box.bottom) || !std::isfinite(padding) || padding <= 0.0F ||
      box.right <= box.left || box.bottom <= box.top) {
    return Status{ErrorCode::invalid_data, "Person box is invalid"};
  }
  PoseRegion region{(box.left + box.right) * 0.5F, (box.top + box.bottom) * 0.5F,
                    (box.right - box.left) * padding, (box.bottom - box.top) * padding};
  constexpr auto aspect_ratio = 192.0F / 256.0F;
  if (region.width > region.height * aspect_ratio) {
    region.height = region.width / aspect_ratio;
  } else {
    region.width = region.height * aspect_ratio;
  }
  return region;
}

Result<std::vector<pose::ModelLandmark>> decode_simcc(std::span<const float> simcc_x,
                                                      std::span<const float> simcc_y,
                                                      std::size_t keypoint_count,
                                                      std::size_t x_bins, std::size_t y_bins,
                                                      PoseRegion region, float split_ratio) {
  if (keypoint_count == 0 || x_bins == 0 || y_bins == 0 ||
      simcc_x.size() != keypoint_count * x_bins || simcc_y.size() != keypoint_count * y_bins ||
      !std::isfinite(split_ratio) || split_ratio <= 0.0F || !std::isfinite(region.center_x) ||
      !std::isfinite(region.center_y) || !std::isfinite(region.width) ||
      !std::isfinite(region.height) || region.width <= 0.0F || region.height <= 0.0F) {
    return Status{ErrorCode::invalid_data, "SimCC tensors or pose region are invalid"};
  }
  std::vector<pose::ModelLandmark> decoded(keypoint_count);
  for (std::size_t keypoint = 0; keypoint < keypoint_count; ++keypoint) {
    const auto x_values = simcc_x.subspan(keypoint * x_bins, x_bins);
    const auto y_values = simcc_y.subspan(keypoint * y_bins, y_bins);
    if (!std::ranges::all_of(x_values, [](float value) { return std::isfinite(value); }) ||
        !std::ranges::all_of(y_values, [](float value) { return std::isfinite(value); })) {
      return Status{ErrorCode::invalid_data, "SimCC tensors contain non-finite values"};
    }
    const auto x_max = std::max_element(x_values.begin(), x_values.end());
    const auto y_max = std::max_element(y_values.begin(), y_values.end());
    if (!std::isfinite(*x_max) || !std::isfinite(*y_max)) {
      decoded[keypoint].confidence = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    const auto model_x = static_cast<float>(std::distance(x_values.begin(), x_max)) / split_ratio;
    const auto model_y = static_cast<float>(std::distance(y_values.begin(), y_max)) / split_ratio;
    decoded[keypoint] = {model_x / 192.0F * region.width + region.center_x - region.width * 0.5F,
                         model_y / 256.0F * region.height + region.center_y - region.height * 0.5F,
                         (*x_max + *y_max) * 0.5F};
  }
  return decoded;
}

}
