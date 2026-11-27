#include "squatris/core/inference/tracking_assessment.hpp"

#include <cmath>

namespace squatris::inference {

namespace {

bool valid_policy(const TrackingPolicy& policy) {
  return policy.maximum_pose_age >= std::chrono::milliseconds::zero() &&
         policy.maximum_capture_gap >= std::chrono::milliseconds::zero() &&
         std::isfinite(policy.minimum_landmark_confidence) &&
         policy.minimum_landmark_confidence >= 0.0F && policy.minimum_landmark_confidence <= 1.0F &&
         policy.minimum_usable_landmarks > 0;
}

}

TrackingAssessment assess_tracking(const TrackingInput& input, const TrackingPolicy& policy) {
  TrackingAssessment assessment;
  if (!valid_policy(policy)) {
    assessment.reason = TrackingReason::inference_failure;
    return assessment;
  }
  if (input.source_error.has_value()) {
    assessment.reason = TrackingReason::source_failure;
    return assessment;
  }
  if (input.now == pose::TimePoint{} ||
      (input.last_capture_observed_at.has_value() && *input.last_capture_observed_at > input.now)) {
    assessment.reason = TrackingReason::invalid_timestamp;
    return assessment;
  }
  if (!input.last_capture_observed_at.has_value() ||
      input.now - *input.last_capture_observed_at > policy.maximum_capture_gap) {
    assessment.reason = TrackingReason::capture_stalled;
    return assessment;
  }
  if (input.inference_error != nullptr) {
    assessment.reason = TrackingReason::inference_failure;
    return assessment;
  }
  if (input.result == nullptr) {
    assessment.reason = TrackingReason::no_result;
    return assessment;
  }
  if (!input.result->person_detected) {
    assessment.reason = TrackingReason::no_person;
    return assessment;
  }
  assessment.state = TrackingState::person_detected;
  if (!input.result->pose.has_value()) {
    assessment.reason = TrackingReason::pose_missing;
    return assessment;
  }
  const auto& pose_frame = *input.result->pose;
  assessment.pose_sequence = pose_frame.sequence;
  assessment.sequence_matches = pose_frame.sequence == input.displayed_frame_sequence;
  if (pose_frame.captured_at == pose::TimePoint{} ||
      pose_frame.inference_completed_at == pose::TimePoint{} ||
      pose_frame.captured_at > input.now || pose_frame.inference_completed_at > input.now ||
      pose_frame.inference_completed_at < pose_frame.captured_at) {
    assessment.state = TrackingState::unavailable;
    assessment.reason = TrackingReason::invalid_timestamp;
    return assessment;
  }
  assessment.pose_age =
      std::chrono::duration_cast<std::chrono::microseconds>(input.now - pose_frame.captured_at);
  if (assessment.pose_age > policy.maximum_pose_age) {
    assessment.state = TrackingState::unavailable;
    assessment.reason = TrackingReason::stale_pose;
    return assessment;
  }
  for (const auto& landmark : pose_frame.landmarks) {
    if (landmark.availability == pose::LandmarkAvailability::available &&
        landmark.confidence.has_value() &&
        *landmark.confidence >= policy.minimum_landmark_confidence) {
      ++assessment.usable_landmarks;
    }
  }
  if (assessment.usable_landmarks < policy.minimum_usable_landmarks) {
    assessment.reason = TrackingReason::insufficient_landmarks;
    return assessment;
  }
  assessment.state = TrackingState::usable_pose;
  assessment.reason = TrackingReason::none;
  return assessment;
}

const char* to_string(TrackingState state) noexcept {
  switch (state) {
  case TrackingState::unavailable:
    return "unavailable";
  case TrackingState::person_detected:
    return "person-detected";
  case TrackingState::usable_pose:
    return "usable-pose";
  }
  return "unavailable";
}

const char* to_string(TrackingReason reason) noexcept {
  switch (reason) {
  case TrackingReason::none:
    return "none";
  case TrackingReason::no_result:
    return "no-result";
  case TrackingReason::inference_failure:
    return "inference-failure";
  case TrackingReason::no_person:
    return "no-person";
  case TrackingReason::pose_missing:
    return "pose-missing";
  case TrackingReason::insufficient_landmarks:
    return "insufficient-landmarks";
  case TrackingReason::invalid_timestamp:
    return "invalid-timestamp";
  case TrackingReason::stale_pose:
    return "stale-pose";
  case TrackingReason::capture_stalled:
    return "capture-stalled";
  case TrackingReason::source_failure:
    return "source-failure";
  }
  return "no-result";
}

}
