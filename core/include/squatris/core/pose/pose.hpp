#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "squatris/core/result.hpp"

namespace squatris::pose {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

enum class LandmarkId : std::uint8_t {
  nose,
  left_shoulder,
  right_shoulder,
  left_elbow,
  right_elbow,
  left_wrist,
  right_wrist,
  left_hip,
  right_hip,
  left_knee,
  right_knee,
  left_ankle,
  right_ankle,
  left_heel,
  right_heel,
  left_foot_index,
  right_foot_index,
  count
};

constexpr std::size_t landmark_count = static_cast<std::size_t>(LandmarkId::count);

enum class LandmarkAvailability : std::uint8_t { unavailable, available };

struct NormalizedImagePoint {
  float x{};
  float y{};
};

struct EstimatedDepth {
  float value{};
};

struct Landmark {
  LandmarkAvailability availability{LandmarkAvailability::unavailable};
  std::optional<NormalizedImagePoint> image;
  std::optional<EstimatedDepth> estimated_depth;
  std::optional<float> confidence;
};

struct ImageSize {
  std::uint32_t width{};
  std::uint32_t height{};
};

struct PoseFrame {
  std::uint64_t sequence{};
  TimePoint captured_at{};
  TimePoint inference_completed_at{};
  ImageSize image_size{};
  std::array<Landmark, landmark_count> landmarks{};
  std::optional<float> pose_confidence;
  std::optional<std::uint64_t> tracking_id;
  std::string inference_backend;

  [[nodiscard]] const Landmark& at(LandmarkId id) const noexcept;
  [[nodiscard]] Landmark& at(LandmarkId id) noexcept;
};

[[nodiscard]] Result<NormalizedImagePoint> normalize_image_point(float pixel_x, float pixel_y,
                                                                 ImageSize image_size);
[[nodiscard]] Landmark make_landmark(NormalizedImagePoint point, std::optional<float> depth,
                                     float confidence);
[[nodiscard]] Status validate_pose(const PoseFrame& frame);
[[nodiscard]] const char* to_string(LandmarkId id) noexcept;

}
