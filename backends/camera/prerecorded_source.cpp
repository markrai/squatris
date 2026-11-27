#include "squatris/backends/camera/prerecorded_source.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <thread>

namespace squatris::backends::camera {

PrerecordedSource::PrerecordedSource(PrerecordedSourceConfig config) : config_(std::move(config)) {}

PrerecordedSource::~PrerecordedSource() { stop(); }

Status PrerecordedSource::start() {
  std::error_code filesystem_error;
  if (config_.path.empty() || !std::filesystem::is_regular_file(config_.path, filesystem_error) ||
      filesystem_error) {
    return {ErrorCode::not_found, "Prerecorded video does not exist"};
  }
  if (start_attempted_.exchange(true)) {
    return {ErrorCode::initialization_failed, "Prerecorded source instances are single-use"};
  }
  started_ = true;
  try {
    worker_ = std::jthread([this](const std::stop_token& stop_token) { run_safely(stop_token); });
  } catch (const std::exception& exception) {
    started_ = false;
    start_attempted_ = false;
    return {ErrorCode::initialization_failed,
            "Prerecorded worker could not start: " + std::string(exception.what())};
  } catch (...) {
    started_ = false;
    start_attempted_ = false;
    return {ErrorCode::initialization_failed, "Prerecorded worker could not start"};
  }
  return Status::success();
}

void PrerecordedSource::run_safely(const std::stop_token& stop_token) noexcept {
  try {
    run(stop_token);
  } catch (const std::exception& exception) {
    {
      std::lock_guard lock(state_mutex_);
      terminal_error_ = Status{ErrorCode::io_error,
                               "Prerecorded worker failed: " + std::string(exception.what())};
    }
    frames_.close();
  } catch (...) {
    {
      std::lock_guard lock(state_mutex_);
      terminal_error_ =
          Status{ErrorCode::io_error, "Prerecorded worker failed with an unknown error"};
    }
    frames_.close();
  }
}

void PrerecordedSource::stop() noexcept {
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
PrerecordedSource::wait_for_frame(std::uint64_t after_sequence, std::chrono::milliseconds timeout) {
  return frames_.wait_for_newer(after_sequence, timeout);
}

input::CaptureFormat PrerecordedSource::format() const {
  std::lock_guard lock(state_mutex_);
  return format_;
}

input::SourceStatistics PrerecordedSource::statistics() const {
  return {measured_fps_.load(), delivered_.load(), frames_.overwritten_count()};
}

std::optional<Status> PrerecordedSource::terminal_error() const {
  std::lock_guard lock(state_mutex_);
  return terminal_error_;
}

void PrerecordedSource::run(const std::stop_token& stop_token) {
  cv::VideoCapture capture(config_.path.string(), cv::CAP_ANY);
  if (!capture.isOpened()) {
    {
      std::lock_guard lock(state_mutex_);
      terminal_error_ = Status{ErrorCode::io_error, "Prerecorded video could not be opened"};
    }
    frames_.close();
    return;
  }
  const auto file_fps = capture.get(cv::CAP_PROP_FPS) > 0.0 ? capture.get(cv::CAP_PROP_FPS) : 30.0;
  {
    std::lock_guard lock(state_mutex_);
    format_ = {static_cast<std::uint32_t>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
               static_cast<std::uint32_t>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)),
               file_fps,
               file_fps,
               capture.getBackendName(),
               "decoded-BGR8"};
  }
  const auto interval = std::chrono::duration<double>(1.0 / file_fps);
  auto next_frame = pose::Clock::now();
  auto rate_start = next_frame;
  std::uint64_t rate_frames{};
  std::uint64_t sequence{};
  while (!stop_token.stop_requested()) {
    const auto acquisition_started = pose::Clock::now();
    cv::Mat image;
    if (!capture.read(image) || image.empty()) {
      if (config_.loop) {
        capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
        continue;
      }
      {
        std::lock_guard lock(state_mutex_);
        terminal_error_ = Status{ErrorCode::end_of_stream, "Prerecorded video completed"};
      }
      frames_.close();
      return;
    }
    auto frame = std::make_shared<input::Frame>();
    frame->sequence = ++sequence;
    const auto source_position = capture.get(cv::CAP_PROP_POS_FRAMES);
    if (std::isfinite(source_position) && source_position >= 1.0) {
      frame->source_frame_index = static_cast<std::uint64_t>(source_position - 1.0);
    }
    const auto source_timestamp_ms = capture.get(cv::CAP_PROP_POS_MSEC);
    if (std::isfinite(source_timestamp_ms) && source_timestamp_ms >= 0.0) {
      frame->source_media_timestamp =
          std::chrono::microseconds(static_cast<std::int64_t>(source_timestamp_ms * 1000.0));
    }
    frame->width = static_cast<std::uint32_t>(image.cols);
    frame->height = static_cast<std::uint32_t>(image.rows);
    frame->stride = frame->width * 3;
    frame->pixels.resize(static_cast<std::size_t>(frame->stride) * frame->height);
    for (std::uint32_t row = 0; row < frame->height; ++row) {
      auto* destination = frame->pixels.data() + static_cast<std::size_t>(row) * frame->stride;
      std::copy_n(image.ptr<std::uint8_t>(static_cast<int>(row)), frame->stride, destination);
    }
    frame->captured_at = pose::Clock::now();
    frame->acquisition_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        frame->captured_at - acquisition_started);
    frames_.publish(std::move(frame));
    delivered_ = sequence;
    ++rate_frames;
    const auto now = pose::Clock::now();
    const auto seconds = std::chrono::duration<double>(now - rate_start).count();
    if (seconds >= 1.0) {
      measured_fps_ = static_cast<double>(rate_frames) / seconds;
      rate_frames = 0;
      rate_start = now;
    }
    if (config_.pace_realtime) {
      next_frame += std::chrono::duration_cast<pose::Clock::duration>(interval);
      std::this_thread::sleep_until(next_frame);
    }
  }
}

}
