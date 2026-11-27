#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "squatris/core/inference/pose_estimator.hpp"

namespace squatris::inference {

enum class TrackingState { unavailable, person_detected, usable_pose };

enum class TrackingReason {
  none,
  no_result,
  inference_failure,
  no_person,
  pose_missing,
  insufficient_landmarks,
  invalid_timestamp,
  stale_pose,
  capture_stalled,
  source_failure
};

struct TrackingPolicy {
  std::chrono::milliseconds maximum_pose_age{250};
  std::chrono::milliseconds maximum_capture_gap{500};
  float minimum_landmark_confidence{0.5F};
  std::size_t minimum_usable_landmarks{3};
};

struct TrackingInput {
  const InferenceResult* result{};
  const Status* inference_error{};
  std::optional<pose::TimePoint> last_capture_observed_at;
  std::optional<Status> source_error;
  std::uint64_t displayed_frame_sequence{};
  pose::TimePoint now{};
};

struct TrackingAssessment {
  TrackingState state{TrackingState::unavailable};
  TrackingReason reason{TrackingReason::no_result};
  std::chrono::microseconds pose_age{};
  std::size_t usable_landmarks{};
  std::uint64_t pose_sequence{};
  bool sequence_matches{};
};

[[nodiscard]] TrackingAssessment assess_tracking(const TrackingInput& input,
                                                 const TrackingPolicy& policy);
[[nodiscard]] const char* to_string(TrackingState state) noexcept;
[[nodiscard]] const char* to_string(TrackingReason reason) noexcept;

}
