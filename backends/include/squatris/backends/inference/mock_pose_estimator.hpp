#pragma once

#include "squatris/core/inference/pose_estimator.hpp"

namespace squatris::backends::inference {

class MockPoseEstimator final : public squatris::inference::IPoseEstimator {
public:
  [[nodiscard]] Result<squatris::inference::InferenceResult>
  infer(const input::Frame& frame) override;
  [[nodiscard]] std::string backend_name() const override;
  [[nodiscard]] std::string execution_provider() const override;
  [[nodiscard]] std::string execution_policy() const override;
};

}
