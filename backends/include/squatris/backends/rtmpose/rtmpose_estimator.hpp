#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "squatris/backends/rtmpose/provider_audit.hpp"
#include "squatris/core/inference/pose_estimator.hpp"

namespace squatris::backends::rtmpose {

enum class ExecutionProvider { cpu, cuda_strict, cuda_audited };

struct RtmposeConfig {
  std::filesystem::path detector_model;
  std::filesystem::path pose_model;
  ExecutionProvider provider{ExecutionProvider::cuda_strict};
  int cuda_device_id{};
  float detection_threshold{0.5F};
  float landmark_threshold{0.2F};
  std::filesystem::path profiling_output_prefix;
  std::filesystem::path production_profiling_output_prefix;
  std::filesystem::path provider_policy_manifest;
  bool verbose_provider_diagnostics{};
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
  [[nodiscard]] std::string execution_policy() const override;
  [[nodiscard]] Result<std::vector<ProviderProfile>> finalize_provider_profiling();
  [[nodiscard]] Result<std::vector<ProviderProfile>> finalize_production_profiling();
  [[nodiscard]] Result<nlohmann::json> provider_policy_evidence() const;

private:
  struct Impl;
  explicit RtmposeEstimator(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}
