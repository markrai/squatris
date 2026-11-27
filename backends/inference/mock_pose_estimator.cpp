#include "squatris/backends/inference/mock_pose_estimator.hpp"

#include <array>
#include <chrono>

namespace squatris::backends::inference {

Result<squatris::inference::InferenceResult> MockPoseEstimator::infer(const input::Frame& frame) {
  const auto started = pose::Clock::now();
  pose::PoseFrame output;
  output.sequence = frame.sequence;
  output.captured_at = frame.captured_at;
  output.image_size = {frame.width, frame.height};
  output.inference_backend = "mock-pose-v1";
  output.pose_confidence = 0.99F;
  output.tracking_id = 1;
  const auto set = [&output](pose::LandmarkId id, float x, float y) {
    output.at(id) = pose::make_landmark({x, y}, std::nullopt, 0.99F);
  };
  set(pose::LandmarkId::nose, 0.50F, 0.15F);
  set(pose::LandmarkId::left_shoulder, 0.42F, 0.27F);
  set(pose::LandmarkId::right_shoulder, 0.58F, 0.27F);
  set(pose::LandmarkId::left_elbow, 0.36F, 0.40F);
  set(pose::LandmarkId::right_elbow, 0.64F, 0.40F);
  set(pose::LandmarkId::left_wrist, 0.32F, 0.53F);
  set(pose::LandmarkId::right_wrist, 0.68F, 0.53F);
  set(pose::LandmarkId::left_hip, 0.45F, 0.53F);
  set(pose::LandmarkId::right_hip, 0.55F, 0.53F);
  set(pose::LandmarkId::left_knee, 0.45F, 0.72F);
  set(pose::LandmarkId::right_knee, 0.55F, 0.72F);
  set(pose::LandmarkId::left_ankle, 0.44F, 0.91F);
  set(pose::LandmarkId::right_ankle, 0.56F, 0.91F);
  set(pose::LandmarkId::left_heel, 0.43F, 0.94F);
  set(pose::LandmarkId::right_heel, 0.57F, 0.94F);
  set(pose::LandmarkId::left_foot_index, 0.47F, 0.95F);
  set(pose::LandmarkId::right_foot_index, 0.61F, 0.95F);
  output.inference_completed_at = pose::Clock::now();
  squatris::inference::InferenceResult result;
  result.pose = std::move(output);
  result.backend = backend_name();
  result.execution_provider = execution_provider();
  result.person_detected = true;
  result.timings.frame_acquisition = frame.acquisition_duration;
  result.timings.frame_age_at_start =
      std::chrono::duration_cast<std::chrono::microseconds>(started - frame.captured_at);
  result.timings.end_to_end =
      std::chrono::duration_cast<std::chrono::microseconds>(pose::Clock::now() - started);
  return result;
}

std::string MockPoseEstimator::backend_name() const { return "mock-pose-v1"; }

std::string MockPoseEstimator::execution_provider() const { return "deterministic-cpu"; }

}
