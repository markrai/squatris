#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>

#include "squatris/core/input/frame_source.hpp"

namespace squatris::backends::camera {

struct SequentialSourceConfig {
  std::filesystem::path video_path;
  std::uint32_t synthetic_width{640};
  std::uint32_t synthetic_height{480};
  double synthetic_fps{60.0};
  bool loop{};
};

class SequentialFrameSource final : public input::IFrameSource {
public:
  explicit SequentialFrameSource(SequentialSourceConfig config = {});
  ~SequentialFrameSource() override;

  [[nodiscard]] Status start() override;
  void stop() noexcept override;
  [[nodiscard]] Result<std::shared_ptr<const input::Frame>>
  wait_for_frame(std::uint64_t after_sequence, std::chrono::milliseconds timeout) override;
  [[nodiscard]] input::CaptureFormat format() const override;
  [[nodiscard]] input::SourceStatistics statistics() const override;
  [[nodiscard]] std::optional<Status> terminal_error() const override;

private:
  struct Impl;

  [[nodiscard]] Result<std::shared_ptr<const input::Frame>> next_frame();

  SequentialSourceConfig config_;
  std::unique_ptr<Impl> implementation_;
  mutable std::mutex mutex_;
  input::CaptureFormat format_{};
  std::optional<Status> terminal_error_;
  std::uint64_t delivered_{};
  std::uint64_t source_loop_index_{};
  std::uint64_t frames_in_loop_{};
  bool start_attempted_{};
  bool started_{};
};

}
