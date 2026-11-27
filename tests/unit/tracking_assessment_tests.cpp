#include <chrono>

#include <gtest/gtest.h>

#include "squatris/core/inference/tracking_assessment.hpp"

namespace {

using namespace std::chrono_literals;

squatris::inference::InferenceResult result_with_landmarks(squatris::pose::TimePoint captured_at,
                                                           std::uint64_t sequence,
                                                           std::size_t landmark_count,
                                                           float confidence = 0.9F) {
  squatris::pose::PoseFrame pose;
  pose.sequence = sequence;
  pose.captured_at = captured_at;
  pose.inference_completed_at = captured_at;
  pose.image_size = {640, 480};
  pose.inference_backend = "failure-injection-fixture";
  for (std::size_t index = 0; index < landmark_count && index < pose.landmarks.size(); ++index) {
    pose.landmarks[index] = squatris::pose::make_landmark({0.5F, 0.5F}, std::nullopt, confidence);
  }
  squatris::inference::InferenceResult result;
  result.person_detected = true;
  result.pose = std::move(pose);
  return result;
}

TEST(TrackingAssessment, ReportsUsablePoseOnlyWhileFresh) {
  const auto captured = squatris::pose::Clock::now();
  const auto result = result_with_landmarks(captured, 7, 4);
  const auto assessment = squatris::inference::assess_tracking(
      {&result, nullptr, captured + 20ms, std::nullopt, 7, captured + 40ms}, {});
  EXPECT_EQ(assessment.state, squatris::inference::TrackingState::usable_pose);
  EXPECT_EQ(assessment.reason, squatris::inference::TrackingReason::none);
  EXPECT_TRUE(assessment.sequence_matches);
  EXPECT_EQ(assessment.usable_landmarks, 4);

  const auto stale = squatris::inference::assess_tracking(
      {&result, nullptr, captured + 300ms, std::nullopt, 7, captured + 300ms}, {});
  EXPECT_EQ(stale.state, squatris::inference::TrackingState::unavailable);
  EXPECT_EQ(stale.reason, squatris::inference::TrackingReason::stale_pose);
}

TEST(TrackingAssessment, DistinguishesDetectionFromUsablePose) {
  const auto now = squatris::pose::Clock::now();
  squatris::inference::InferenceResult detected;
  detected.person_detected = true;
  const auto no_pose =
      squatris::inference::assess_tracking({&detected, nullptr, now, std::nullopt, 3, now}, {});
  EXPECT_EQ(no_pose.state, squatris::inference::TrackingState::person_detected);
  EXPECT_EQ(no_pose.reason, squatris::inference::TrackingReason::pose_missing);

  const auto weak = result_with_landmarks(now, 3, 2, 0.1F);
  const auto insufficient =
      squatris::inference::assess_tracking({&weak, nullptr, now, std::nullopt, 3, now}, {});
  EXPECT_EQ(insufficient.state, squatris::inference::TrackingState::person_detected);
  EXPECT_EQ(insufficient.reason, squatris::inference::TrackingReason::insufficient_landmarks);
}

TEST(TrackingAssessment, FailureInjectionInvalidatesPreviousEvidence) {
  const auto now = squatris::pose::Clock::now();
  const auto previous = result_with_landmarks(now, 4, 4);
  const squatris::Status injected{squatris::ErrorCode::inference_failed,
                                  "deterministic injected failure"};
  const auto failed = squatris::inference::assess_tracking(
      {&previous, &injected, now, std::nullopt, 4, now + 1ms}, {});
  EXPECT_EQ(failed.state, squatris::inference::TrackingState::unavailable);
  EXPECT_EQ(failed.reason, squatris::inference::TrackingReason::inference_failure);
}

TEST(TrackingAssessment, CaptureStallAndSourceFailureInvalidateEvidence) {
  const auto now = squatris::pose::Clock::now();
  const auto result = result_with_landmarks(now, 9, 4);
  const auto stalled = squatris::inference::assess_tracking(
      {&result, nullptr, now, std::nullopt, 9, now + 600ms}, {});
  EXPECT_EQ(stalled.reason, squatris::inference::TrackingReason::capture_stalled);
  EXPECT_EQ(stalled.state, squatris::inference::TrackingState::unavailable);

  const squatris::Status disconnected{squatris::ErrorCode::disconnected, "injected disconnect"};
  const auto failed = squatris::inference::assess_tracking(
      {&result, nullptr, now + 1ms, disconnected, 9, now + 2ms}, {});
  EXPECT_EQ(failed.reason, squatris::inference::TrackingReason::source_failure);
  EXPECT_EQ(failed.state, squatris::inference::TrackingState::unavailable);
}

TEST(TrackingAssessment, ReportsFrameMismatchWithoutRelabelingItCurrent) {
  const auto now = squatris::pose::Clock::now();
  const auto result = result_with_landmarks(now, 10, 4);
  const auto assessment = squatris::inference::assess_tracking(
      {&result, nullptr, now, std::nullopt, 12, now + 1ms}, {});
  EXPECT_EQ(assessment.state, squatris::inference::TrackingState::usable_pose);
  EXPECT_FALSE(assessment.sequence_matches);
  EXPECT_EQ(assessment.pose_sequence, 10);
}

TEST(TrackingAssessment, RejectsMalformedAndFutureTimestamps) {
  const auto now = squatris::pose::Clock::now();
  auto future_capture = result_with_landmarks(now + 1ms, 11, 4);
  const auto future = squatris::inference::assess_tracking(
      {&future_capture, nullptr, now, std::nullopt, 11, now}, {});
  EXPECT_EQ(future.state, squatris::inference::TrackingState::unavailable);
  EXPECT_EQ(future.reason, squatris::inference::TrackingReason::invalid_timestamp);

  auto future_completion = result_with_landmarks(now - 20ms, 12, 4);
  future_completion.pose->inference_completed_at = now + 1ms;
  const auto completion = squatris::inference::assess_tracking(
      {&future_completion, nullptr, now, std::nullopt, 12, now}, {});
  EXPECT_EQ(completion.reason, squatris::inference::TrackingReason::invalid_timestamp);

  auto reversed = result_with_landmarks(now - 20ms, 13, 4);
  reversed.pose->inference_completed_at = now - 21ms;
  const auto malformed =
      squatris::inference::assess_tracking({&reversed, nullptr, now, std::nullopt, 13, now}, {});
  EXPECT_EQ(malformed.reason, squatris::inference::TrackingReason::invalid_timestamp);

  const auto future_observation = squatris::inference::assess_tracking(
      {&reversed, nullptr, now + 1ms, std::nullopt, 13, now}, {});
  EXPECT_EQ(future_observation.reason, squatris::inference::TrackingReason::invalid_timestamp);
}

}
