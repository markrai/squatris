#include "squatris/backends/camera/sequential_frame_source.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace squatris::backends::camera {

struct SequentialFrameSource::Impl {
  cv::VideoCapture capture;
};

SequentialFrameSource::SequentialFrameSource(SequentialSourceConfig config)
    : config_(std::move(config)) {}

SequentialFrameSource::~SequentialFrameSource() { stop(); }

Status SequentialFrameSource::start() {
  std::lock_guard lock(mutex_);
  if (start_attempted_) {
    return {ErrorCode::initialization_failed, "Sequential source instances are single-use"};
  }
  if (config_.video_path.empty() &&
      (config_.synthetic_width == 0 || config_.synthetic_height == 0 ||
       !std::isfinite(config_.synthetic_fps) || config_.synthetic_fps <= 0.0)) {
    return {ErrorCode::invalid_argument, "Sequential synthetic configuration is invalid"};
  }
  std::error_code filesystem_error;
  if (!config_.video_path.empty() &&
      (!std::filesystem::is_regular_file(config_.video_path, filesystem_error) ||
       filesystem_error)) {
    return {ErrorCode::not_found, "Sequential video does not exist"};
  }
  start_attempted_ = true;
  try {
    implementation_ = std::make_unique<Impl>();
    if (config_.video_path.empty()) {
      format_ = {config_.synthetic_width, config_.synthetic_height,  config_.synthetic_fps,
                 config_.synthetic_fps,   "deterministic-synthetic", "BGR8"};
      started_ = true;
      return Status::success();
    }
    implementation_->capture.open(config_.video_path.string(), cv::CAP_ANY);
    if (!implementation_->capture.isOpened()) {
      implementation_.reset();
      start_attempted_ = false;
      return {ErrorCode::io_error, "Sequential video could not be opened"};
    }
    const auto source_fps = implementation_->capture.get(cv::CAP_PROP_FPS);
    const auto fps = std::isfinite(source_fps) && source_fps > 0.0 ? source_fps : 30.0;
    format_ = {static_cast<std::uint32_t>(implementation_->capture.get(cv::CAP_PROP_FRAME_WIDTH)),
               static_cast<std::uint32_t>(implementation_->capture.get(cv::CAP_PROP_FRAME_HEIGHT)),
               fps,
               fps,
               implementation_->capture.getBackendName(),
               "decoded-BGR8-sequential"};
    started_ = true;
    return Status::success();
  } catch (const std::exception& exception) {
    implementation_.reset();
    start_attempted_ = false;
    return {ErrorCode::io_error,
            "Sequential source startup failed: " + std::string(exception.what())};
  } catch (...) {
    implementation_.reset();
    start_attempted_ = false;
    return {ErrorCode::io_error, "Sequential source startup failed with an unknown error"};
  }
}

void SequentialFrameSource::stop() noexcept {
  std::lock_guard lock(mutex_);
  started_ = false;
  implementation_.reset();
}

Result<std::shared_ptr<const input::Frame>>
SequentialFrameSource::wait_for_frame(std::uint64_t after_sequence,
                                      std::chrono::milliseconds timeout) {
  static_cast<void>(timeout);
  std::lock_guard lock(mutex_);
  if (!started_) {
    return Status{ErrorCode::stopped, "Sequential source is stopped"};
  }
  if (after_sequence != delivered_) {
    return Status{ErrorCode::invalid_argument,
                  "Sequential source requires exact in-order frame consumption"};
  }
  Result<std::shared_ptr<const input::Frame>> frame =
      Status{ErrorCode::io_error, "Sequential source failed before reading a frame"};
  try {
    frame = next_frame();
  } catch (const std::exception& exception) {
    frame =
        Status{ErrorCode::io_error, "Sequential source failed: " + std::string(exception.what())};
  } catch (...) {
    frame = Status{ErrorCode::io_error, "Sequential source failed with an unknown error"};
  }
  if (!frame) {
    terminal_error_ = frame.error();
    started_ = false;
  }
  return frame;
}

input::CaptureFormat SequentialFrameSource::format() const {
  std::lock_guard lock(mutex_);
  return format_;
}

input::SourceStatistics SequentialFrameSource::statistics() const {
  std::lock_guard lock(mutex_);
  return {0.0, delivered_, 0};
}

std::optional<Status> SequentialFrameSource::terminal_error() const {
  std::lock_guard lock(mutex_);
  return terminal_error_;
}

Result<std::shared_ptr<const input::Frame>> SequentialFrameSource::next_frame() {
  const auto acquisition_started = pose::Clock::now();
  cv::Mat image;
  std::optional<std::uint64_t> source_index;
  std::optional<std::chrono::microseconds> media_timestamp;
  if (config_.video_path.empty()) {
    image.create(static_cast<int>(config_.synthetic_height),
                 static_cast<int>(config_.synthetic_width), CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
      auto* row = image.ptr<cv::Vec3b>(y);
      for (int x = 0; x < image.cols; ++x) {
        row[x][0] = static_cast<std::uint8_t>((static_cast<std::uint64_t>(x) + delivered_) % 256);
        row[x][1] = static_cast<std::uint8_t>((static_cast<std::uint64_t>(y) + delivered_) % 256);
        row[x][2] = static_cast<std::uint8_t>((x + y) % 256);
      }
    }
    source_index = delivered_;
    media_timestamp = std::chrono::microseconds(static_cast<std::int64_t>(
        static_cast<double>(delivered_) * 1'000'000.0 / config_.synthetic_fps));
  } else {
    try {
      if (!implementation_->capture.read(image) || image.empty()) {
        return Status{ErrorCode::end_of_stream, "Sequential video completed"};
      }
      const auto position = implementation_->capture.get(cv::CAP_PROP_POS_FRAMES);
      if (std::isfinite(position) && position >= 1.0) {
        source_index = static_cast<std::uint64_t>(position - 1.0);
      }
      const auto timestamp_ms = implementation_->capture.get(cv::CAP_PROP_POS_MSEC);
      if (std::isfinite(timestamp_ms) && timestamp_ms >= 0.0) {
        media_timestamp =
            std::chrono::microseconds(static_cast<std::int64_t>(timestamp_ms * 1000.0));
      }
    } catch (const std::exception& exception) {
      return Status{ErrorCode::io_error,
                    "Sequential video read failed: " + std::string(exception.what())};
    }
  }
  if (image.type() != CV_8UC3) {
    return Status{ErrorCode::invalid_data, "Sequential source did not deliver BGR8 frames"};
  }
  auto frame = std::make_shared<input::Frame>();
  frame->sequence = ++delivered_;
  frame->source_frame_index = source_index;
  frame->source_media_timestamp = media_timestamp;
  frame->width = static_cast<std::uint32_t>(image.cols);
  frame->height = static_cast<std::uint32_t>(image.rows);
  frame->stride = frame->width * 3;
  frame->pixels.resize(static_cast<std::size_t>(frame->stride) * frame->height);
  for (std::uint32_t row = 0; row < frame->height; ++row) {
    const auto* begin = image.ptr<std::uint8_t>(static_cast<int>(row));
    auto* destination = frame->pixels.data() + static_cast<std::size_t>(row) * frame->stride;
    std::copy_n(begin, frame->stride, destination);
  }
  frame->captured_at = pose::Clock::now();
  frame->acquisition_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      frame->captured_at - acquisition_started);
  return std::shared_ptr<const input::Frame>(std::move(frame));
}

}
