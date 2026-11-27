#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>

#include "squatris/core/input/frame_source.hpp"
#include "squatris/core/input/latest_frame_buffer.hpp"

namespace squatris::backends::camera {

struct PrerecordedSourceConfig {
  std::filesystem::path path;
  bool pace_realtime{true};
  bool loop{};
};

class PrerecordedSource final : public input::IFrameSource {
public:
  explicit PrerecordedSource(PrerecordedSourceConfig config);
  ~PrerecordedSource() override;

  [[nodiscard]] Status start() override;
  void stop() noexcept override;
  [[nodiscard]] Result<std::shared_ptr<const input::Frame>>
  wait_for_frame(std::uint64_t after_sequence, std::chrono::milliseconds timeout) override;
  [[nodiscard]] input::CaptureFormat format() const override;
  [[nodiscard]] input::SourceStatistics statistics() const override;
  [[nodiscard]] std::optional<Status> terminal_error() const override;

private:
  void run(const std::stop_token& stop_token);

  PrerecordedSourceConfig config_;
  input::LatestFrameBuffer frames_;
  std::jthread worker_;
  std::atomic<bool> started_{};
  std::atomic<std::uint64_t> delivered_{};
  std::atomic<double> measured_fps_{};
  mutable std::mutex state_mutex_;
  input::CaptureFormat format_{};
  std::optional<Status> terminal_error_;
};

}
