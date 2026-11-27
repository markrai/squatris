#include "squatris/backends/camera/opencv_camera_source.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <ranges>
#include <sstream>

namespace squatris::backends::camera {

namespace {

int capture_backend() {
#ifdef _WIN32
  return cv::CAP_MSMF;
#else
  return cv::CAP_ANY;
#endif
}

std::string fourcc_string(double value) {
  const auto code = static_cast<int>(value);
  std::string text(4, ' ');
  for (int index = 0; index < 4; ++index) {
    text[static_cast<std::size_t>(index)] = static_cast<char>((code >> (8 * index)) & 0xFF);
  }
  const auto printable = std::ranges::all_of(
      text, [](unsigned char character) { return character >= 0x20 && character <= 0x7E; });
  if (printable) {
    return text;
  }
  std::ostringstream formatted;
  formatted << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0')
            << static_cast<std::uint32_t>(code);
  return formatted.str();
}

input::CaptureFormat read_format(cv::VideoCapture& capture, double requested_fps) {
  return {static_cast<std::uint32_t>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
          static_cast<std::uint32_t>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)),
          requested_fps,
          capture.get(cv::CAP_PROP_FPS),
          capture.getBackendName(),
          "BGR8 (source " + fourcc_string(capture.get(cv::CAP_PROP_FOURCC)) + ")"};
}

void request_format(cv::VideoCapture& capture, std::uint32_t width, std::uint32_t height,
                    double fps) {
  capture.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(width));
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height));
  capture.set(cv::CAP_PROP_FPS, fps);
}

}

OpenCvCameraSource::OpenCvCameraSource(CameraConfig config) : config_(config) {}

OpenCvCameraSource::~OpenCvCameraSource() { stop(); }

Result<std::vector<CameraDevice>> OpenCvCameraSource::enumerate(bool probe_formats) {
  constexpr std::array probes{std::array{1920.0, 1080.0, 60.0}, std::array{1920.0, 1080.0, 30.0},
                              std::array{1280.0, 720.0, 60.0},  std::array{1280.0, 720.0, 30.0},
                              std::array{640.0, 480.0, 60.0},   std::array{640.0, 480.0, 30.0}};
  std::vector<CameraDevice> devices;
  for (int index = 0; index < 10; ++index) {
    cv::VideoCapture capture(index, capture_backend());
    if (!capture.isOpened()) {
      continue;
    }
    CameraDevice device{index, "camera-" + std::to_string(index), {}};
    if (probe_formats) {
      for (const auto& probe : probes) {
        request_format(capture, static_cast<std::uint32_t>(probe[0]),
                       static_cast<std::uint32_t>(probe[1]), probe[2]);
        cv::Mat sample;
        if (!capture.read(sample) || sample.empty()) {
          continue;
        }
        auto actual = read_format(capture, probe[2]);
        const auto duplicate =
            std::ranges::any_of(device.accepted_probe_formats, [&actual](const auto& existing) {
              return existing.width == actual.width && existing.height == actual.height &&
                     std::abs(existing.negotiated_fps - actual.negotiated_fps) < 0.1;
            });
        if (!duplicate) {
          device.accepted_probe_formats.push_back(std::move(actual));
        }
      }
    }
    devices.push_back(std::move(device));
  }
  return devices;
}

Status OpenCvCameraSource::start() {
  if (started_.exchange(true)) {
    return {ErrorCode::initialization_failed, "Camera source is already started"};
  }
  if (config_.device_index < 0 || config_.width == 0 || config_.height == 0 ||
      !std::isfinite(config_.fps) || config_.fps <= 0.0) {
    started_ = false;
    return {ErrorCode::invalid_argument, "Camera configuration is invalid"};
  }
  worker_ = std::jthread([this](const std::stop_token& stop_token) { run(stop_token); });
  return Status::success();
}

void OpenCvCameraSource::stop() noexcept {
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
OpenCvCameraSource::wait_for_frame(std::uint64_t after_sequence,
                                   std::chrono::milliseconds timeout) {
  return frames_.wait_for_newer(after_sequence, timeout);
}

input::CaptureFormat OpenCvCameraSource::format() const {
  std::lock_guard lock(state_mutex_);
  return negotiated_;
}

input::SourceStatistics OpenCvCameraSource::statistics() const {
  return {measured_fps_.load(), delivered_.load(), frames_.overwritten_count()};
}

std::optional<Status> OpenCvCameraSource::terminal_error() const {
  std::lock_guard lock(state_mutex_);
  return terminal_error_;
}

void OpenCvCameraSource::run(const std::stop_token& stop_token) {
  cv::VideoCapture capture(config_.device_index, capture_backend());
  if (!capture.isOpened()) {
    {
      std::lock_guard lock(state_mutex_);
      terminal_error_ = Status{ErrorCode::unavailable, "Camera could not be opened"};
    }
    frames_.close();
    return;
  }
  request_format(capture, config_.width, config_.height, config_.fps);
  {
    std::lock_guard lock(state_mutex_);
    negotiated_ = read_format(capture, config_.fps);
  }
  auto rate_start = pose::Clock::now();
  std::uint64_t rate_frames{};
  std::uint64_t sequence{};
  while (!stop_token.stop_requested()) {
    const auto acquisition_started = pose::Clock::now();
    cv::Mat image;
    if (!capture.read(image) || image.empty()) {
      {
        std::lock_guard lock(state_mutex_);
        terminal_error_ = Status{ErrorCode::disconnected, "Camera stopped delivering frames"};
      }
      frames_.close();
      return;
    }
    if (image.type() != CV_8UC3) {
      {
        std::lock_guard lock(state_mutex_);
        terminal_error_ = Status{ErrorCode::invalid_data, "Camera did not deliver BGR8 frames"};
      }
      frames_.close();
      return;
    }
    auto frame = std::make_shared<input::Frame>();
    frame->sequence = ++sequence;
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
  }
}

}
