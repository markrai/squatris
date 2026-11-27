#include "squatris/backends/camera/synthetic_source.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <thread>

namespace squatris::backends::camera {

SyntheticSource::SyntheticSource(SyntheticSourceConfig config) : config_(config) {}

SyntheticSource::~SyntheticSource() { stop(); }

Status SyntheticSource::start() {
  if (started_.exchange(true)) {
    return {ErrorCode::initialization_failed, "Synthetic source is already started"};
  }
  if (config_.width == 0 || config_.height == 0 || !std::isfinite(config_.fps) ||
      config_.fps <= 0.0) {
    started_ = false;
    return {ErrorCode::invalid_argument, "Synthetic source configuration is invalid"};
  }
  worker_ = std::jthread([this](const std::stop_token& stop_token) { run(stop_token); });
  return Status::success();
}

void SyntheticSource::stop() noexcept {
  if (!started_.exchange(false)) {
    return;
  }
  worker_.request_stop();
  frames_.close();
  if (worker_.joinable()) {
    worker_.join();
  }
}

Result<std::shared_ptr<const input::Frame>>
SyntheticSource::wait_for_frame(std::uint64_t after_sequence, std::chrono::milliseconds timeout) {
  return frames_.wait_for_newer(after_sequence, timeout);
}

input::CaptureFormat SyntheticSource::format() const {
  return {config_.width, config_.height, config_.fps, config_.fps, "synthetic", "BGR8"};
}

input::SourceStatistics SyntheticSource::statistics() const {
  return {measured_fps_.load(), delivered_.load(), frames_.overwritten_count()};
}

std::optional<Status> SyntheticSource::terminal_error() const {
  std::lock_guard lock(error_mutex_);
  return terminal_error_;
}

void SyntheticSource::run(const std::stop_token& stop_token) {
  const auto interval = std::chrono::duration<double>(1.0 / config_.fps);
  auto next_frame = pose::Clock::now();
  auto rate_start = next_frame;
  std::uint64_t rate_frames{};
  std::uint64_t sequence{};
  while (!stop_token.stop_requested()) {
    if (config_.fail_after_frames && sequence >= *config_.fail_after_frames) {
      {
        std::lock_guard lock(error_mutex_);
        terminal_error_ = Status{ErrorCode::disconnected, "Synthetic disconnection requested"};
      }
      frames_.close();
      return;
    }
    const auto acquisition_started = pose::Clock::now();
    auto frame = std::make_shared<input::Frame>();
    frame->sequence = ++sequence;
    frame->width = config_.width;
    frame->height = config_.height;
    frame->stride = config_.width * 3;
    frame->pixels.resize(static_cast<std::size_t>(frame->stride) * frame->height);
    for (std::uint32_t y = 0; y < frame->height; ++y) {
      for (std::uint32_t x = 0; x < frame->width; ++x) {
        const auto offset = static_cast<std::size_t>(y) * frame->stride + x * 3;
        frame->pixels[offset] = static_cast<std::uint8_t>((x + sequence) % 256);
        frame->pixels[offset + 1] = static_cast<std::uint8_t>((y + sequence) % 256);
        frame->pixels[offset + 2] = static_cast<std::uint8_t>((x + y) % 256);
      }
    }
    frame->captured_at = pose::Clock::now();
    frame->acquisition_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        frame->captured_at - acquisition_started);
    frames_.publish(std::move(frame));
    delivered_ = sequence;
    ++rate_frames;
    const auto now = pose::Clock::now();
    const auto rate_duration = std::chrono::duration<double>(now - rate_start).count();
    if (rate_duration >= 1.0) {
      measured_fps_ = static_cast<double>(rate_frames) / rate_duration;
      rate_frames = 0;
      rate_start = now;
    }
    next_frame += std::chrono::duration_cast<pose::Clock::duration>(interval);
    std::this_thread::sleep_until(next_frame);
  }
}

}
