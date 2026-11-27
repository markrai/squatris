#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "squatris/backends/camera/opencv_camera_source.hpp"
#include "squatris/backends/camera/prerecorded_source.hpp"
#include "squatris/backends/camera/synthetic_source.hpp"
#include "squatris/backends/inference/mock_pose_estimator.hpp"
#include "squatris/backends/rtmpose/rtmpose_estimator.hpp"
#include "squatris/core/metrics/latency_statistics.hpp"
#include "squatris/core/pose/geometry.hpp"

namespace {

using squatris::ErrorCode;
using squatris::Status;
using squatris::inference::InferenceResult;
using squatris::pose::LandmarkId;

struct Options {
  bool help{};
  bool list_cameras{};
  bool synthetic{};
  int camera{};
  std::uint32_t width{1920};
  std::uint32_t height{1080};
  double fps{60.0};
  std::filesystem::path video;
  std::string backend{"rtmpose"};
  std::string provider{"cuda"};
  std::filesystem::path detector{"models/detector.onnx"};
  std::filesystem::path pose{"models/pose.onnx"};
};

struct SharedState {
  std::mutex mutex;
  std::optional<InferenceResult> latest;
  std::optional<Status> error;
  std::uint64_t processed{};
  std::uint64_t skipped{};
  double inference_fps{};
};

std::optional<std::string> value_after(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    return std::nullopt;
  }
  return argv[++index];
}

template <typename Value> bool parse_number(const std::string& text, Value& output) {
  const auto* end = text.data() + text.size();
  const auto result = std::from_chars(text.data(), end, output);
  return result.ec == std::errc{} && result.ptr == end;
}

std::optional<Options> parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      options.help = true;
    } else if (argument == "--list-cameras") {
      options.list_cameras = true;
    } else if (argument == "--synthetic") {
      options.synthetic = true;
    } else {
      const auto value = value_after(index, argc, argv);
      if (!value) {
        std::cerr << "Missing value for " << argument << '\n';
        return std::nullopt;
      }
      const auto numeric_valid = [&] {
        if (argument == "--camera") {
          return parse_number(*value, options.camera);
        }
        if (argument == "--width") {
          return parse_number(*value, options.width);
        }
        if (argument == "--height") {
          return parse_number(*value, options.height);
        }
        if (argument == "--fps") {
          return parse_number(*value, options.fps);
        }
        return true;
      }();
      if (!numeric_valid) {
        std::cerr << "Invalid value for " << argument << '\n';
        return std::nullopt;
      }
      if (argument == "--video") {
        options.video = *value;
      } else if (argument == "--backend") {
        options.backend = *value;
      } else if (argument == "--provider") {
        options.provider = *value;
      } else if (argument == "--detector") {
        options.detector = *value;
      } else if (argument == "--pose") {
        options.pose = *value;
      } else if (argument != "--camera" && argument != "--width" && argument != "--height" &&
                 argument != "--fps") {
        std::cerr << "Unknown argument: " << argument << '\n';
        return std::nullopt;
      }
    }
  }
  return options;
}

void print_help() {
  std::cout << "Squatris Pose Lab\n"
            << "  --list-cameras\n"
            << "  --camera INDEX --width WIDTH --height HEIGHT --fps FPS\n"
            << "  --video PATH | --synthetic\n"
            << "  --backend rtmpose|mock --provider cuda|cpu\n"
            << "  --detector PATH --pose PATH\n"
            << "Keys: Q/Esc exit, S skeleton, D diagnostics, M mirror preview\n";
}

std::unique_ptr<squatris::input::IFrameSource> make_source(const Options& options) {
  if (options.synthetic) {
    return std::make_unique<squatris::backends::camera::SyntheticSource>(
        squatris::backends::camera::SyntheticSourceConfig{options.width, options.height,
                                                          options.fps});
  }
  if (!options.video.empty()) {
    return std::make_unique<squatris::backends::camera::PrerecordedSource>(
        squatris::backends::camera::PrerecordedSourceConfig{options.video, true, true});
  }
  return std::make_unique<squatris::backends::camera::OpenCvCameraSource>(
      squatris::backends::camera::CameraConfig{options.camera, options.width, options.height,
                                               options.fps});
}

squatris::Result<std::unique_ptr<squatris::inference::IPoseEstimator>>
make_estimator(const Options& options) {
  if (options.backend == "mock") {
    return std::unique_ptr<squatris::inference::IPoseEstimator>(
        std::make_unique<squatris::backends::inference::MockPoseEstimator>());
  }
  if (options.backend != "rtmpose") {
    return Status{ErrorCode::invalid_argument, "Backend must be rtmpose or mock"};
  }
  squatris::backends::rtmpose::RtmposeConfig config;
  config.detector_model = options.detector;
  config.pose_model = options.pose;
  if (options.provider == "cuda") {
    config.provider = squatris::backends::rtmpose::ExecutionProvider::cuda;
  } else if (options.provider == "cpu") {
    config.provider = squatris::backends::rtmpose::ExecutionProvider::cpu;
  } else {
    return Status{ErrorCode::invalid_argument, "Provider must be cuda or cpu"};
  }
  auto estimator = squatris::backends::rtmpose::RtmposeEstimator::create(config);
  if (!estimator) {
    return estimator.error();
  }
  return std::unique_ptr<squatris::inference::IPoseEstimator>(estimator.take_value().release());
}

cv::Point display_point(const squatris::pose::Landmark& landmark, const cv::Mat& image,
                        bool mirror) {
  const auto point = landmark.image.value_or(squatris::pose::NormalizedImagePoint{});
  auto x = point.x;
  if (mirror) {
    x = 1.0F - x;
  }
  return {static_cast<int>(x * static_cast<float>(image.cols)),
          static_cast<int>(point.y * static_cast<float>(image.rows))};
}

bool available(const squatris::pose::Landmark& landmark) {
  return landmark.availability == squatris::pose::LandmarkAvailability::available &&
         landmark.image.has_value() && landmark.confidence.has_value();
}

void draw_pose(cv::Mat& image, const squatris::pose::PoseFrame& pose, bool mirror) {
  constexpr std::array connections{std::pair{LandmarkId::left_shoulder, LandmarkId::right_shoulder},
                                   std::pair{LandmarkId::left_shoulder, LandmarkId::left_elbow},
                                   std::pair{LandmarkId::left_elbow, LandmarkId::left_wrist},
                                   std::pair{LandmarkId::right_shoulder, LandmarkId::right_elbow},
                                   std::pair{LandmarkId::right_elbow, LandmarkId::right_wrist},
                                   std::pair{LandmarkId::left_shoulder, LandmarkId::left_hip},
                                   std::pair{LandmarkId::right_shoulder, LandmarkId::right_hip},
                                   std::pair{LandmarkId::left_hip, LandmarkId::right_hip},
                                   std::pair{LandmarkId::left_hip, LandmarkId::left_knee},
                                   std::pair{LandmarkId::left_knee, LandmarkId::left_ankle},
                                   std::pair{LandmarkId::right_hip, LandmarkId::right_knee},
                                   std::pair{LandmarkId::right_knee, LandmarkId::right_ankle},
                                   std::pair{LandmarkId::left_ankle, LandmarkId::left_heel},
                                   std::pair{LandmarkId::left_heel, LandmarkId::left_foot_index},
                                   std::pair{LandmarkId::right_ankle, LandmarkId::right_heel},
                                   std::pair{LandmarkId::right_heel, LandmarkId::right_foot_index}};
  constexpr float reliable = 0.5F;
  for (const auto& [first_id, second_id] : connections) {
    const auto& first = pose.at(first_id);
    const auto& second = pose.at(second_id);
    if (!available(first) || !available(second)) {
      continue;
    }
    const auto confidence = std::min(*first.confidence, *second.confidence);
    const auto color = confidence >= reliable ? cv::Scalar(60, 220, 60) : cv::Scalar(0, 165, 255);
    cv::line(image, display_point(first, image, mirror), display_point(second, image, mirror),
             color, 2, cv::LINE_AA);
  }
  for (std::size_t index = 0; index < pose.landmarks.size(); ++index) {
    const auto& landmark = pose.landmarks[index];
    if (!available(landmark)) {
      continue;
    }
    const auto color =
        *landmark.confidence >= reliable ? cv::Scalar(80, 255, 80) : cv::Scalar(0, 165, 255);
    const auto point = display_point(landmark, image, mirror);
    cv::circle(image, point, 5, color, cv::FILLED, cv::LINE_AA);
    std::ostringstream confidence;
    confidence << std::fixed << std::setprecision(2) << *landmark.confidence;
    cv::putText(image, confidence.str(), point + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.35,
                color, 1, cv::LINE_AA);
  }
}

std::string angle_text(std::optional<float> angle) {
  if (!angle) {
    return "unavailable";
  }
  std::ostringstream text;
  text << std::fixed << std::setprecision(1) << *angle << " deg";
  return text.str();
}

void draw_diagnostics(cv::Mat& image, const squatris::input::CaptureFormat& format,
                      const squatris::input::SourceStatistics& source_statistics,
                      const SharedState& state, const std::optional<InferenceResult>& inference,
                      const std::optional<Status>& error, bool mirror) {
  std::vector<std::string> lines;
  std::ostringstream format_line;
  format_line << format.width << 'x' << format.height << " requested " << format.requested_fps
              << " FPS negotiated " << format.negotiated_fps << " FPS " << format.backend << ' '
              << format.encoding;
  lines.push_back(format_line.str());
  std::ostringstream rates;
  rates << std::fixed << std::setprecision(1) << "capture " << source_statistics.delivered_fps
        << " FPS  inference " << state.inference_fps << " FPS";
  lines.push_back(rates.str());
  lines.push_back("preview mirror: " + std::string(mirror ? "on" : "off") +
                  "  skipped: " + std::to_string(state.skipped) +
                  "  overwritten: " + std::to_string(source_statistics.frames_overwritten));
  if (inference) {
    lines.push_back("backend: " + inference->backend +
                    "  provider: " + inference->execution_provider);
    std::ostringstream timing;
    timing << std::fixed << std::setprecision(2) << "acquire "
           << static_cast<double>(inference->timings.frame_acquisition.count()) / 1000.0
           << " ms  latency " << static_cast<double>(inference->timings.end_to_end.count()) / 1000.0
           << " ms  frame age "
           << static_cast<double>(inference->timings.frame_age_at_start.count()) / 1000.0
           << " ms  detector "
           << static_cast<double>(inference->timings.person_detection.count()) / 1000.0
           << " ms  pose GPU/CPU "
           << static_cast<double>(inference->timings.pose_inference.count()) / 1000.0 << " ms";
    lines.push_back(timing.str());
    if (inference->pose) {
      const auto& pose = *inference->pose;
      const auto left = squatris::pose::joint_angle_degrees(pose.at(LandmarkId::left_hip),
                                                            pose.at(LandmarkId::left_knee),
                                                            pose.at(LandmarkId::left_ankle), 0.5F);
      const auto right = squatris::pose::joint_angle_degrees(
          pose.at(LandmarkId::right_hip), pose.at(LandmarkId::right_knee),
          pose.at(LandmarkId::right_ankle), 0.5F);
      lines.push_back("tracking: valid  left knee: " + angle_text(left) +
                      "  right knee: " + angle_text(right));
    } else {
      lines.push_back("tracking: no person");
    }
  }
  if (error) {
    lines.push_back("error: " + error->message);
  }
  const auto overlay_height = static_cast<int>(lines.size()) * 24 + 12;
  cv::Mat overlay = image(cv::Rect(0, 0, image.cols, std::min(overlay_height, image.rows)));
  cv::Mat shade(overlay.size(), overlay.type(), cv::Scalar(0, 0, 0));
  cv::addWeighted(shade, 0.65, overlay, 0.35, 0.0, overlay);
  int y = 24;
  for (const auto& line : lines) {
    cv::putText(image, line, {10, y}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(240, 240, 240), 1,
                cv::LINE_AA);
    y += 24;
  }
}

void inference_loop(const std::stop_token& stop_token, squatris::input::IFrameSource& source,
                    squatris::inference::IPoseEstimator& estimator, SharedState& state) {
  std::uint64_t sequence{};
  auto rate_started = squatris::pose::Clock::now();
  std::uint64_t rate_count{};
  while (!stop_token.stop_requested()) {
    auto frame = source.wait_for_frame(sequence, std::chrono::milliseconds(100));
    if (!frame) {
      if (frame.error().code == ErrorCode::timeout) {
        continue;
      }
      if (stop_token.stop_requested() || frame.error().code == ErrorCode::stopped) {
        return;
      }
      std::lock_guard lock(state.mutex);
      state.error = frame.error();
      return;
    }
    if (sequence != 0 && frame.value()->sequence > sequence + 1) {
      std::lock_guard lock(state.mutex);
      state.skipped += frame.value()->sequence - sequence - 1;
    }
    sequence = frame.value()->sequence;
    auto inference = estimator.infer(*frame.value());
    if (!inference) {
      std::lock_guard lock(state.mutex);
      state.error = inference.error();
      continue;
    }
    ++rate_count;
    const auto now = squatris::pose::Clock::now();
    const auto seconds = std::chrono::duration<double>(now - rate_started).count();
    std::lock_guard lock(state.mutex);
    state.latest = inference.take_value();
    ++state.processed;
    if (seconds >= 1.0) {
      state.inference_fps = static_cast<double>(rate_count) / seconds;
      rate_count = 0;
      rate_started = now;
    }
  }
}

int list_cameras() {
  auto devices = squatris::backends::camera::OpenCvCameraSource::enumerate(true);
  if (!devices) {
    std::cerr << devices.error().message << '\n';
    return 1;
  }
  if (devices.value().empty()) {
    std::cout << "No OpenCV Media Foundation cameras found.\n";
    return 0;
  }
  for (const auto& device : devices.value()) {
    std::cout << '[' << device.index << "] " << device.name << '\n';
    for (const auto& format : device.accepted_probe_formats) {
      std::cout << "  " << format.width << 'x' << format.height << " @ " << format.negotiated_fps
                << " FPS " << format.backend << ' ' << format.encoding << '\n';
    }
  }
  return 0;
}

}

int main(int argc, char** argv) {
  const auto options = parse_options(argc, argv);
  if (!options) {
    print_help();
    return 2;
  }
  if (options->help) {
    print_help();
    return 0;
  }
  if (options->list_cameras) {
    return list_cameras();
  }
  auto estimator = make_estimator(*options);
  if (!estimator) {
    std::cerr << "Inference initialization failed: " << estimator.error().message << '\n';
    return 1;
  }
  auto source = make_source(*options);
  const auto start_status = source->start();
  if (!start_status.ok()) {
    std::cerr << "Capture initialization failed: " << start_status.message << '\n';
    return 1;
  }
  SharedState state;
  std::jthread inference_worker([&](const std::stop_token& stop_token) {
    inference_loop(stop_token, *source, *estimator.value(), state);
  });
  bool skeleton = true;
  bool diagnostics = true;
  bool mirror = true;
  std::uint64_t display_sequence{};
  cv::namedWindow("Squatris Pose Lab", cv::WINDOW_NORMAL);
  while (true) {
    auto frame = source->wait_for_frame(display_sequence, std::chrono::milliseconds(20));
    if (frame) {
      display_sequence = frame.value()->sequence;
      cv::Mat wrapped(
          static_cast<int>(frame.value()->height), static_cast<int>(frame.value()->width), CV_8UC3,
          const_cast<std::uint8_t*>(frame.value()->pixels.data()), frame.value()->stride);
      cv::Mat display;
      if (mirror) {
        cv::flip(wrapped, display, 1);
      } else {
        display = wrapped.clone();
      }
      std::optional<InferenceResult> latest;
      std::optional<Status> error;
      SharedState snapshot;
      {
        std::lock_guard lock(state.mutex);
        latest = state.latest;
        error = state.error;
        snapshot.processed = state.processed;
        snapshot.skipped = state.skipped;
        snapshot.inference_fps = state.inference_fps;
      }
      if (skeleton && latest && latest->pose) {
        draw_pose(display, *latest->pose, mirror);
      }
      if (diagnostics) {
        draw_diagnostics(display, source->format(), source->statistics(), snapshot, latest, error,
                         mirror);
      }
      cv::imshow("Squatris Pose Lab", display);
    } else if (frame.error().code != ErrorCode::timeout) {
      const auto terminal = source->terminal_error();
      if (terminal) {
        std::cerr << "Capture ended: " << terminal->message << '\n';
      } else if (frame.error().code != ErrorCode::stopped) {
        std::cerr << "Capture failed: " << frame.error().message << '\n';
      }
      break;
    }
    const auto key = cv::waitKey(1) & 0xFF;
    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }
    if (key == 's' || key == 'S') {
      skeleton = !skeleton;
    } else if (key == 'd' || key == 'D') {
      diagnostics = !diagnostics;
    } else if (key == 'm' || key == 'M') {
      mirror = !mirror;
    }
  }
  inference_worker.request_stop();
  source->stop();
  if (inference_worker.joinable()) {
    inference_worker.join();
  }
  cv::destroyAllWindows();
  return 0;
}
