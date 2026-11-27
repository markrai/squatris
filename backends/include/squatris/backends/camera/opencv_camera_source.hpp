#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "squatris/core/input/frame_source.hpp"
#include "squatris/core/input/latest_frame_buffer.hpp"

namespace squatris::backends::camera {

struct CameraConfig {
  int device_index{};
  std::uint32_t width{1920};
  std::uint32_t height{1080};
  double fps{60.0};
};

struct CameraDevice {
  int index{};
  std::string name;
  std::vector<input::CaptureFormat> accepted_probe_formats;
};

class OpenCvCameraSource final : public input::IFrameSource {
public:
  explicit OpenCvCameraSource(CameraConfig config);
  ~OpenCvCameraSource() override;

  [[nodiscard]] static Result<std::vector<CameraDevice>> enumerate(bool probe_formats);
  [[nodiscard]] Status start() override;
  void stop() noexcept override;
  [[nodiscard]] Result<std::shared_ptr<const input::Frame>>
  wait_for_frame(std::uint64_t after_sequence, std::chrono::milliseconds timeout) override;
  [[nodiscard]] input::CaptureFormat format() const override;
  [[nodiscard]] input::SourceStatistics statistics() const override;
  [[nodiscard]] std::optional<Status> terminal_error() const override;

private:
  void run_safely(const std::stop_token& stop_token) noexcept;
  void run(const std::stop_token& stop_token);

  CameraConfig config_;
  input::LatestFrameBuffer frames_;
  std::jthread worker_;
  std::atomic<bool> started_{};
  std::atomic<bool> start_attempted_{};
  std::atomic<std::uint64_t> delivered_{};
  std::atomic<double> measured_fps_{};
  mutable std::mutex state_mutex_;
  input::CaptureFormat negotiated_{};
  std::optional<Status> terminal_error_;
};

}
