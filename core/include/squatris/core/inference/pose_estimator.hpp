#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "squatris/core/input/frame.hpp"
#include "squatris/core/pose/pose.hpp"
#include "squatris/core/result.hpp"

namespace squatris::inference {

struct InferenceTimings {
  std::chrono::microseconds frame_acquisition{};
  std::chrono::microseconds frame_age_at_start{};
  std::chrono::microseconds person_detection{};
  std::chrono::microseconds pose_preprocessing{};
  std::chrono::microseconds pose_inference{};
  std::chrono::microseconds pose_postprocessing{};
  std::chrono::microseconds end_to_end{};
};

struct InferenceResult {
  std::optional<pose::PoseFrame> pose;
  InferenceTimings timings;
  std::string backend;
  std::string execution_provider;
  bool person_detected{};
};

class IPoseEstimator {
public:
  virtual ~IPoseEstimator() = default;
  [[nodiscard]] virtual Result<InferenceResult> infer(const input::Frame& frame) = 0;
  [[nodiscard]] virtual std::string backend_name() const = 0;
  [[nodiscard]] virtual std::string execution_provider() const = 0;
};

}
