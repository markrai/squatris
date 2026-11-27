#pragma once

#include <filesystem>
#include <memory>

#include "squatris/core/inference/pose_estimator.hpp"

namespace squatris::backends::rtmpose {

enum class ExecutionProvider { cpu, cuda };

struct RtmposeConfig {
  std::filesystem::path detector_model;
  std::filesystem::path pose_model;
  ExecutionProvider provider{ExecutionProvider::cuda};
  int cuda_device_id{};
  float detection_threshold{0.5F};
  float landmark_threshold{0.2F};
};

class RtmposeEstimator final : public squatris::inference::IPoseEstimator {
public:
  ~RtmposeEstimator() override;
  RtmposeEstimator(RtmposeEstimator&&) noexcept;
  RtmposeEstimator& operator=(RtmposeEstimator&&) noexcept;
  RtmposeEstimator(const RtmposeEstimator&) = delete;
  RtmposeEstimator& operator=(const RtmposeEstimator&) = delete;

  [[nodiscard]] static Result<std::unique_ptr<RtmposeEstimator>>
  create(const RtmposeConfig& config);
  [[nodiscard]] Result<squatris::inference::InferenceResult>
  infer(const input::Frame& frame) override;
  [[nodiscard]] std::string backend_name() const override;
  [[nodiscard]] std::string execution_provider() const override;

private:
  struct Impl;
  explicit RtmposeEstimator(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}
