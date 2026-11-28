#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "squatris/backends/rtmpose/provider_audit.hpp"
#include "squatris/core/inference/pose_estimator.hpp"

namespace squatris::backends::rtmpose {

enum class ExecutionProvider { cpu, cuda_strict, cuda_audited };

enum class PoseModel { rtmpose_m_halpe26, rtmw_m_wholebody133 };

[[nodiscard]] constexpr std::string_view pose_model_name(PoseModel model) noexcept {
  return model == PoseModel::rtmw_m_wholebody133 ? "rtmw-m-coco133-256x192"
                                                : "rtmpose-m-halpe26-256x192";
}

[[nodiscard]] constexpr std::size_t pose_model_keypoint_count(PoseModel model) noexcept {
  return model == PoseModel::rtmw_m_wholebody133 ? 133U : 26U;
}

struct ModelExecutionTimings {
  std::chrono::microseconds detector_inference{};
  std::chrono::microseconds pose_inference{};
  std::chrono::microseconds total{};
};

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
  bool recreate_audited_sessions_for_experiment{};
  PoseModel pose_variant{PoseModel::rtmpose_m_halpe26};
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
  [[nodiscard]] Result<ModelExecutionTimings> infer_synthetic_tensors();

private:
  struct Impl;
  explicit RtmposeEstimator(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}
