#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/version.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include "sha256.hpp"
#include "squatris/backends/camera/prerecorded_source.hpp"
#include "squatris/backends/camera/sequential_frame_source.hpp"
#include "squatris/backends/camera/synthetic_source.hpp"
#include "squatris/backends/inference/mock_pose_estimator.hpp"
#include "squatris/backends/rtmpose/rtmpose_estimator.hpp"
#include "squatris/core/inference/tracking_assessment.hpp"
#include "squatris/core/metrics/latency_statistics.hpp"

namespace {

using squatris::ErrorCode;
using squatris::Status;

enum class BenchmarkMode : std::uint8_t { deterministic, realtime };

struct Options {
  std::filesystem::path video;
  std::filesystem::path output{"benchmark-result.json"};
  std::filesystem::path detector{"models/detector.onnx"};
  std::filesystem::path pose{"models/pose.onnx"};
  std::filesystem::path manifest{"models/manifest.json"};
  std::filesystem::path validate_report;
  std::string backend{"mock"};
  std::string provider{"cpu"};
  BenchmarkMode mode{BenchmarkMode::deterministic};
  std::size_t warmup{20};
  std::size_t iterations{200};
  std::size_t maximum_consecutive_failures{3};
  std::uint32_t width{640};
  std::uint32_t height{480};
  std::uint32_t deadline_seconds{600};
  std::uint32_t frame_timeout_ms{5000};
};

struct PhaseStatistics {
  explicit PhaseStatistics(std::size_t capacity)
      : end_to_end(capacity), acquisition(capacity), frame_age(capacity), detector(capacity),
        detector_preprocessing(capacity), detector_inference(capacity),
        detector_postprocessing(capacity), preprocessing(capacity), inference(capacity),
        postprocessing(capacity), capture_to_completion(capacity) {
    frame_records.reserve(capacity);
  }

  squatris::metrics::LatencyStatistics end_to_end;
  squatris::metrics::LatencyStatistics acquisition;
  squatris::metrics::LatencyStatistics frame_age;
  squatris::metrics::LatencyStatistics detector;
  squatris::metrics::LatencyStatistics detector_preprocessing;
  squatris::metrics::LatencyStatistics detector_inference;
  squatris::metrics::LatencyStatistics detector_postprocessing;
  squatris::metrics::LatencyStatistics preprocessing;
  squatris::metrics::LatencyStatistics inference;
  squatris::metrics::LatencyStatistics postprocessing;
  squatris::metrics::LatencyStatistics capture_to_completion;
  std::vector<nlohmann::json> frame_records;
  std::optional<squatris::pose::TimePoint> started_at;
  squatris::pose::TimePoint completed_at{};
  std::uint64_t no_pose{};
  std::uint64_t usable_pose{};
  std::uint64_t full_body_complete{};
  std::array<std::uint64_t, 6> left_landmark_available{};
  std::array<std::uint64_t, 6> right_landmark_available{};
  std::array<std::uint64_t, 6> pair_available{};
};

struct BodyRegion {
  std::string_view name;
  squatris::pose::LandmarkId left;
  squatris::pose::LandmarkId right;
};

constexpr std::array body_regions{
    BodyRegion{"shoulders", squatris::pose::LandmarkId::left_shoulder,
               squatris::pose::LandmarkId::right_shoulder},
    BodyRegion{"elbows", squatris::pose::LandmarkId::left_elbow,
               squatris::pose::LandmarkId::right_elbow},
    BodyRegion{"wrists", squatris::pose::LandmarkId::left_wrist,
               squatris::pose::LandmarkId::right_wrist},
    BodyRegion{"hips", squatris::pose::LandmarkId::left_hip, squatris::pose::LandmarkId::right_hip},
    BodyRegion{"knees", squatris::pose::LandmarkId::left_knee,
               squatris::pose::LandmarkId::right_knee},
    BodyRegion{"ankles", squatris::pose::LandmarkId::left_ankle,
               squatris::pose::LandmarkId::right_ankle}};

std::optional<std::string> next_value(int& index, int argc, char** argv) {
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

std::optional<Options> parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto value = next_value(index, argc, argv);
    if (!value) {
      std::cerr << "Missing value for " << argument << '\n';
      return std::nullopt;
    }
    const auto numeric_valid = [&] {
      if (argument == "--warmup") {
        return parse_number(*value, options.warmup);
      }
      if (argument == "--iterations") {
        return parse_number(*value, options.iterations);
      }
      if (argument == "--width") {
        return parse_number(*value, options.width);
      }
      if (argument == "--height") {
        return parse_number(*value, options.height);
      }
      if (argument == "--deadline-seconds") {
        return parse_number(*value, options.deadline_seconds);
      }
      if (argument == "--frame-timeout-ms") {
        return parse_number(*value, options.frame_timeout_ms);
      }
      if (argument == "--max-consecutive-failures") {
        return parse_number(*value, options.maximum_consecutive_failures);
      }
      return true;
    }();
    if (!numeric_valid) {
      std::cerr << "Invalid value for " << argument << '\n';
      return std::nullopt;
    }
    if (argument == "--video") {
      options.video = *value;
    } else if (argument == "--output") {
      options.output = *value;
    } else if (argument == "--detector") {
      options.detector = *value;
    } else if (argument == "--pose") {
      options.pose = *value;
    } else if (argument == "--manifest") {
      options.manifest = *value;
    } else if (argument == "--validate-report") {
      options.validate_report = *value;
    } else if (argument == "--backend") {
      options.backend = *value;
    } else if (argument == "--provider") {
      options.provider = *value;
    } else if (argument == "--mode") {
      if (*value == "deterministic") {
        options.mode = BenchmarkMode::deterministic;
      } else if (*value == "realtime") {
        options.mode = BenchmarkMode::realtime;
      } else {
        std::cerr << "Benchmark mode must be deterministic or realtime\n";
        return std::nullopt;
      }
    } else if (argument != "--warmup" && argument != "--iterations" && argument != "--width" &&
               argument != "--height" && argument != "--deadline-seconds" &&
               argument != "--frame-timeout-ms" && argument != "--max-consecutive-failures") {
      std::cerr << "Unknown argument: " << argument << '\n';
      return std::nullopt;
    }
  }
  if (!options.validate_report.empty()) {
    return options;
  }
  if (options.iterations == 0 || options.width == 0 || options.height == 0 ||
      options.deadline_seconds == 0 || options.frame_timeout_ms == 0 ||
      options.maximum_consecutive_failures == 0) {
    std::cerr << "Iterations, dimensions, time limits, and failure bounds must be nonzero\n";
    return std::nullopt;
  }
  if (options.backend != "mock" && options.backend != "rtmpose") {
    std::cerr << "Backend must be mock or rtmpose\n";
    return std::nullopt;
  }
  if (options.provider != "cpu" && options.provider != "cuda") {
    std::cerr << "Provider must be cpu or cuda\n";
    return std::nullopt;
  }
  return options;
}

std::unique_ptr<squatris::input::IFrameSource> make_source(const Options& options) {
  if (options.mode == BenchmarkMode::deterministic) {
    return std::make_unique<squatris::backends::camera::SequentialFrameSource>(
        squatris::backends::camera::SequentialSourceConfig{options.video, options.width,
                                                           options.height, 60.0});
  }
  if (!options.video.empty()) {
    return std::make_unique<squatris::backends::camera::PrerecordedSource>(
        squatris::backends::camera::PrerecordedSourceConfig{options.video, true, false});
  }
  return std::make_unique<squatris::backends::camera::SyntheticSource>(
      squatris::backends::camera::SyntheticSourceConfig{options.width, options.height, 60.0});
}

squatris::Result<std::unique_ptr<squatris::inference::IPoseEstimator>>
make_estimator(const Options& options) {
  if (options.backend == "mock") {
    return std::unique_ptr<squatris::inference::IPoseEstimator>(
        std::make_unique<squatris::backends::inference::MockPoseEstimator>());
  }
  squatris::backends::rtmpose::RtmposeConfig config;
  config.detector_model = options.detector;
  config.pose_model = options.pose;
  config.provider = options.provider == "cuda"
                        ? squatris::backends::rtmpose::ExecutionProvider::cuda
                        : squatris::backends::rtmpose::ExecutionProvider::cpu;
  if (config.provider == squatris::backends::rtmpose::ExecutionProvider::cuda) {
    config.profiling_output_prefix =
        options.output.parent_path() / (options.output.stem().string() + "-ort-profile");
  }
  auto estimator = squatris::backends::rtmpose::RtmposeEstimator::create(config);
  if (!estimator) {
    return estimator.error();
  }
  return std::unique_ptr<squatris::inference::IPoseEstimator>(estimator.take_value().release());
}

std::string environment_value(const char* name) {
#ifdef _WIN32
  char* value{};
  std::size_t length{};
  if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
    return "unknown";
  }
  std::string result(value);
  std::free(value);
  return result;
#else
  const auto* value = std::getenv(name);
  return value != nullptr ? value : "unknown";
#endif
}

std::vector<std::string> display_adapters() {
  std::vector<std::string> adapters;
#ifdef _WIN32
  for (DWORD index = 0;; ++index) {
    DISPLAY_DEVICEA device{};
    device.cb = sizeof(device);
    if (EnumDisplayDevicesA(nullptr, index, &device, 0) == 0) {
      break;
    }
    if ((device.StateFlags & DISPLAY_DEVICE_MIRRORING_DRIVER) == 0) {
      adapters.emplace_back(device.DeviceString);
    }
  }
#endif
  return adapters;
}

std::uint64_t physical_memory_bytes() {
#ifdef _WIN32
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status) != 0) {
    return status.ullTotalPhys;
  }
#endif
  return 0;
}

std::string compiler_version() {
#ifdef _MSC_VER
  return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
  return "Clang " + std::string(__clang_version__);
#elif defined(__GNUC__)
  return "GCC " + std::string(__VERSION__);
#else
  return "unknown";
#endif
}

std::string error_code_name(ErrorCode code) {
  switch (code) {
  case ErrorCode::none:
    return "none";
  case ErrorCode::invalid_argument:
    return "invalid_argument";
  case ErrorCode::invalid_data:
    return "invalid_data";
  case ErrorCode::not_found:
    return "not_found";
  case ErrorCode::unavailable:
    return "unavailable";
  case ErrorCode::timeout:
    return "timeout";
  case ErrorCode::end_of_stream:
    return "end_of_stream";
  case ErrorCode::disconnected:
    return "disconnected";
  case ErrorCode::initialization_failed:
    return "initialization_failed";
  case ErrorCode::inference_failed:
    return "inference_failed";
  case ErrorCode::io_error:
    return "io_error";
  case ErrorCode::stopped:
    return "stopped";
  }
  return "unknown";
}

nlohmann::json percentile_json(const squatris::metrics::Percentiles& percentiles) {
  return {{"samples", percentiles.samples},
          {"median_ms", percentiles.median_ms},
          {"p95_ms", percentiles.p95_ms},
          {"p99_ms", percentiles.p99_ms}};
}

double milliseconds(std::chrono::microseconds duration) {
  return static_cast<double>(duration.count()) / 1000.0;
}

bool usable_landmark(const squatris::pose::Landmark& landmark, float minimum_confidence) {
  return landmark.availability == squatris::pose::LandmarkAvailability::available &&
         landmark.image.has_value() && landmark.confidence.has_value() &&
         std::isfinite(*landmark.confidence) && *landmark.confidence >= minimum_confidence;
}

void add_sample(PhaseStatistics& phase, const squatris::input::Frame& frame,
                const squatris::inference::InferenceResult& result,
                const squatris::inference::TrackingPolicy& tracking_policy) {
  phase.end_to_end.add(result.timings.end_to_end);
  phase.acquisition.add(result.timings.frame_acquisition);
  phase.frame_age.add(result.timings.frame_age_at_start);
  phase.detector.add(result.timings.person_detection);
  phase.detector_preprocessing.add(result.timings.detector_preprocessing);
  phase.detector_inference.add(result.timings.detector_inference);
  phase.detector_postprocessing.add(result.timings.detector_postprocessing);
  phase.preprocessing.add(result.timings.pose_preprocessing);
  phase.inference.add(result.timings.pose_inference);
  phase.postprocessing.add(result.timings.pose_postprocessing);
  phase.capture_to_completion.add(result.timings.frame_age_at_start + result.timings.end_to_end);
  if (!result.pose) {
    ++phase.no_pose;
  }
  const auto assessed_at = squatris::pose::Clock::now();
  const auto tracking = squatris::inference::assess_tracking(
      {&result, nullptr, assessed_at, std::nullopt, frame.sequence, assessed_at}, tracking_policy);
  if (tracking.state == squatris::inference::TrackingState::usable_pose) {
    ++phase.usable_pose;
  }
  nlohmann::json record = {
      {"sequence", frame.sequence},
      {"person_detected", result.person_detected},
      {"pose_available", result.pose.has_value()},
      {"usable_pose", tracking.state == squatris::inference::TrackingState::usable_pose},
      {"tracking_state", squatris::inference::to_string(tracking.state)},
      {"tracking_reason", squatris::inference::to_string(tracking.reason)},
      {"usable_landmarks", tracking.usable_landmarks}};
  nlohmann::json coverage = nlohmann::json::object();
  bool complete = result.pose.has_value();
  for (std::size_t index = 0; index < body_regions.size(); ++index) {
    const auto& region = body_regions[index];
    const auto left =
        result.pose.has_value() &&
        usable_landmark(result.pose->at(region.left), tracking_policy.minimum_landmark_confidence);
    const auto right =
        result.pose.has_value() &&
        usable_landmark(result.pose->at(region.right), tracking_policy.minimum_landmark_confidence);
    phase.left_landmark_available[index] += left ? 1 : 0;
    phase.right_landmark_available[index] += right ? 1 : 0;
    phase.pair_available[index] += left && right ? 1 : 0;
    complete = complete && left && right;
    coverage[region.name] = {{"left", left}, {"right", right}, {"complete", left && right}};
  }
  if (complete) {
    ++phase.full_body_complete;
  }
  record["full_body_complete"] = complete;
  record["landmark_coverage"] = std::move(coverage);
  record["timings_ms"] = {
      {"frame_acquisition", milliseconds(result.timings.frame_acquisition)},
      {"frame_age_at_inference_start", milliseconds(result.timings.frame_age_at_start)},
      {"detector_total", milliseconds(result.timings.person_detection)},
      {"detector_preprocessing", milliseconds(result.timings.detector_preprocessing)},
      {"detector_inference", milliseconds(result.timings.detector_inference)},
      {"detector_postprocessing", milliseconds(result.timings.detector_postprocessing)},
      {"pose_preprocessing", milliseconds(result.timings.pose_preprocessing)},
      {"pose_inference", milliseconds(result.timings.pose_inference)},
      {"pose_postprocessing", milliseconds(result.timings.pose_postprocessing)},
      {"estimator_end_to_end", milliseconds(result.timings.end_to_end)},
      {"capture_delivery_to_pose_completion",
       milliseconds(result.timings.frame_age_at_start + result.timings.end_to_end)}};
  record["source_frame_index"] = frame.source_frame_index
                                     ? nlohmann::json(*frame.source_frame_index)
                                     : nlohmann::json(nullptr);
  record["source_media_timestamp_us"] = frame.source_media_timestamp
                                            ? nlohmann::json(frame.source_media_timestamp->count())
                                            : nlohmann::json(nullptr);
  phase.frame_records.push_back(std::move(record));
  phase.completed_at = squatris::pose::Clock::now();
}

nlohmann::json phase_json(const PhaseStatistics& phase) {
  const auto elapsed =
      phase.started_at
          ? std::chrono::duration<double>(phase.completed_at - *phase.started_at).count()
          : 0.0;
  const auto processed = phase.frame_records.size();
  nlohmann::json coverage = nlohmann::json::object();
  for (std::size_t index = 0; index < body_regions.size(); ++index) {
    coverage[body_regions[index].name] = {
        {"left_available_frames", phase.left_landmark_available[index]},
        {"right_available_frames", phase.right_landmark_available[index]},
        {"complete_frames", phase.pair_available[index]}};
  }
  return {{"frames_processed", processed},
          {"no_pose_frames", phase.no_pose},
          {"pose_available_frames", processed - phase.no_pose},
          {"usable_pose_frames", phase.usable_pose},
          {"full_body_complete_frames", phase.full_body_complete},
          {"elapsed_seconds", elapsed},
          {"throughput_fps", elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0},
          {"useful_pose_throughput_fps",
           elapsed > 0.0 ? static_cast<double>(phase.usable_pose) / elapsed : 0.0},
          {"landmark_coverage", coverage},
          {"processed_frames", phase.frame_records},
          {"timings",
           {{"frame_acquisition", percentile_json(phase.acquisition.calculate())},
            {"estimator_end_to_end", percentile_json(phase.end_to_end.calculate())},
            {"frame_age_at_inference_start", percentile_json(phase.frame_age.calculate())},
            {"detector_total", percentile_json(phase.detector.calculate())},
            {"detector_preprocessing", percentile_json(phase.detector_preprocessing.calculate())},
            {"detector_inference", percentile_json(phase.detector_inference.calculate())},
            {"detector_postprocessing", percentile_json(phase.detector_postprocessing.calculate())},
            {"pose_preprocessing", percentile_json(phase.preprocessing.calculate())},
            {"pose_inference", percentile_json(phase.inference.calculate())},
            {"pose_postprocessing", percentile_json(phase.postprocessing.calculate())},
            {"capture_delivery_to_pose_completion",
             percentile_json(phase.capture_to_completion.calculate())}}}};
}

squatris::Result<nlohmann::json> load_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return Status{ErrorCode::not_found, "Could not open JSON file: " + path.string()};
  }
  try {
    return nlohmann::json::parse(input);
  } catch (const nlohmann::json::exception& exception) {
    return Status{ErrorCode::invalid_data,
                  "Invalid JSON in " + path.string() + ": " + std::string(exception.what())};
  }
}

nlohmann::json hardware_identity() {
  return {{"processor", environment_value("PROCESSOR_IDENTIFIER")},
          {"logical_processors", std::thread::hardware_concurrency()},
          {"physical_memory_bytes", physical_memory_bytes()},
          {"display_adapters", display_adapters()},
          {"computer_name", environment_value("COMPUTERNAME")},
          {"os", environment_value("OS")}};
}

nlohmann::json build_identity() {
  return {{"configuration",
#ifdef NDEBUG
           "Release"
#else
           "Debug"
#endif
          },
          {"compiler", compiler_version()},
          {"opencv", CV_VERSION},
          {"nlohmann_json", std::to_string(NLOHMANN_JSON_VERSION_MAJOR) + "." +
                                std::to_string(NLOHMANN_JSON_VERSION_MINOR) + "." +
                                std::to_string(NLOHMANN_JSON_VERSION_PATCH)},
          {"onnxruntime", SQUATRIS_ONNXRUNTIME_VERSION},
          {"onnxruntime_compiled", static_cast<bool>(SQUATRIS_HAS_ONNXRUNTIME)}};
}

nlohmann::json initial_report(const Options& options) {
  const PhaseStatistics empty(0);
  const auto execution_provider = options.backend == "mock"    ? "deterministic-cpu"
                                  : options.provider == "cuda" ? "CUDAExecutionProvider"
                                                               : "CPUExecutionProvider";
  return {{"schema_version", 3},
          {"status",
           {{"success", false},
            {"error_count", 1},
            {"deadline_exceeded", false},
            {"terminal_error",
             {{"code", "initialization_failed"},
              {"stage", "startup"},
              {"message", "Benchmark did not complete"}}}}},
          {"hardware", hardware_identity()},
          {"build", build_identity()},
          {"model",
           {{"backend", options.backend},
            {"execution_provider", execution_provider},
            {"detector_sha256", nullptr},
            {"pose_sha256", nullptr},
            {"checksums_verified", options.backend == "mock"},
            {"strict_provider_assignment", false},
            {"provider_evidence",
             {{"method", options.provider == "cuda" ? "onnxruntime-profiling" : "not-requested"},
              {"verified", options.backend == "mock" || options.provider != "cuda"},
              {"cuda_graph_execution", false},
              {"profiles", nlohmann::json::array()}}}}},
          {"input",
           {{"kind", options.video.empty() ? "synthetic" : "video"},
            {"source",
             options.video.empty() ? "deterministic-generated-pattern" : options.video.string()},
            {"source_sha256", nullptr},
            {"width", options.width},
            {"height", options.height},
            {"source_fps", 0.0}}},
          {"configuration",
           {{"mode", options.mode == BenchmarkMode::deterministic ? "deterministic" : "realtime"},
            {"warmup_iterations", options.warmup},
            {"measured_iterations", options.iterations},
            {"deadline_seconds", options.deadline_seconds},
            {"frame_timeout_ms", options.frame_timeout_ms},
            {"maximum_consecutive_failures", options.maximum_consecutive_failures},
            {"tracking_policy",
             {{"maximum_pose_age_ms", 250},
              {"maximum_capture_gap_ms", 500},
              {"minimum_landmark_confidence", 0.5},
              {"minimum_usable_landmarks", 3}}}}},
          {"runtime",
           {{"initialization_ms", nullptr},
            {"first_inference_ms", nullptr},
            {"timing_attribution",
             {{"detector_preprocessing", "CPU"},
              {"detector_inference", "ONNX Runtime synchronous execution"},
              {"detector_postprocessing", "CPU"},
              {"pose_preprocessing", "CPU"},
              {"pose_inference", "ONNX Runtime synchronous execution"},
              {"pose_postprocessing", "CPU"},
              {"cuda_graph_execution", false}}}}},
          {"warmup", phase_json(empty)},
          {"measured", phase_json(empty)},
          {"results", {{"frames_skipped", 0}, {"source_frames_delivered", 0}}}};
}

Status write_report(const std::filesystem::path& path, const nlohmann::json& report) {
  try {
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
      return {ErrorCode::io_error, "Could not write " + path.string()};
    }
    output << report.dump(2) << '\n';
    if (!output) {
      return {ErrorCode::io_error, "Could not finish writing " + path.string()};
    }
    return Status::success();
  } catch (const std::exception& exception) {
    return {ErrorCode::io_error,
            "Could not write " + path.string() + ": " + std::string(exception.what())};
  }
}

void record_failure(nlohmann::json& report, const Status& error, std::string_view stage,
                    std::uint64_t errors = 1) {
  report["status"] = {
      {"success", false},
      {"error_count", std::max<std::uint64_t>(errors, 1)},
      {"deadline_exceeded", error.code == ErrorCode::timeout && stage == "execution-deadline"},
      {"terminal_error",
       {{"code", error_code_name(error.code)}, {"stage", stage}, {"message", error.message}}}};
}

squatris::Result<nlohmann::json>
provider_profiling_evidence(squatris::inference::IPoseEstimator& estimator) {
  auto* rtmpose = dynamic_cast<squatris::backends::rtmpose::RtmposeEstimator*>(&estimator);
  if (rtmpose == nullptr) {
    return Status{ErrorCode::invalid_argument,
                  "CUDA provider profiling requires the RTMPose estimator"};
  }
  const auto profiles = rtmpose->finalize_provider_profiling();
  if (!profiles) {
    return profiles.error();
  }
  nlohmann::json profile_records = nlohmann::json::array();
  bool verified = profiles.value().size() == 2;
  for (const auto& profile : profiles.value()) {
    const auto trace = load_json(profile.path);
    if (!trace || !trace.value().is_array()) {
      return trace ? Status{ErrorCode::invalid_data,
                            "ONNX Runtime profile is not an event array: " + profile.path.string()}
                   : trace.error();
    }
    std::uint64_t cuda_nodes{};
    std::uint64_t cpu_nodes{};
    std::uint64_t other_nodes{};
    for (const auto& event : trace.value()) {
      if (!event.is_object() || !event.contains("args") || !event["args"].is_object() ||
          !event["args"].contains("provider") || !event["args"]["provider"].is_string()) {
        continue;
      }
      const auto provider = event["args"]["provider"].get<std::string>();
      if (provider == "CUDAExecutionProvider") {
        ++cuda_nodes;
      } else if (provider == "CPUExecutionProvider") {
        ++cpu_nodes;
      } else {
        ++other_nodes;
      }
    }
    const auto hash = squatris::tools::sha256_file(profile.path);
    if (!hash) {
      return hash.error();
    }
    const auto session_verified = cuda_nodes > 0 && cpu_nodes == 0 && other_nodes == 0;
    verified = verified && session_verified;
    profile_records.push_back({{"session", profile.session},
                               {"path", std::filesystem::absolute(profile.path).string()},
                               {"sha256", hash.value()},
                               {"cuda_node_events", cuda_nodes},
                               {"cpu_node_events", cpu_nodes},
                               {"other_node_events", other_nodes},
                               {"verified", session_verified}});
  }
  return nlohmann::json{{"method", "onnxruntime-profiling"},
                        {"verified", verified},
                        {"cuda_graph_execution", false},
                        {"profiles", profile_records}};
}

squatris::Result<nlohmann::json> verified_model_identity(const Options& options) {
  if (options.backend != "rtmpose") {
    return nlohmann::json{{"backend", "mock-pose-v1"},
                          {"execution_provider", "deterministic-cpu"},
                          {"detector_sha256", nullptr},
                          {"pose_sha256", nullptr},
                          {"checksums_verified", true},
                          {"strict_provider_assignment", false},
                          {"provider_evidence",
                           {{"method", "not-applicable"},
                            {"verified", true},
                            {"cuda_graph_execution", false},
                            {"profiles", nlohmann::json::array()}}}};
  }
  const auto manifest = load_json(options.manifest);
  if (!manifest) {
    return manifest.error();
  }
  if (!manifest.value().contains("artifacts") || !manifest.value()["artifacts"].is_array()) {
    return Status{ErrorCode::invalid_data, "Model manifest has no artifacts array"};
  }
  std::optional<std::string> expected_detector;
  std::optional<std::string> expected_pose;
  for (const auto& artifact : manifest.value()["artifacts"]) {
    if (!artifact.contains("role") || !artifact.contains("onnx_sha256")) {
      continue;
    }
    if (artifact["role"] == "person_detector") {
      expected_detector = artifact["onnx_sha256"].get<std::string>();
    } else if (artifact["role"] == "pose_estimator") {
      expected_pose = artifact["onnx_sha256"].get<std::string>();
    }
  }
  if (!expected_detector || !expected_pose) {
    return Status{ErrorCode::invalid_data, "Model manifest lacks required detector or pose hashes"};
  }
  const auto detector_hash = squatris::tools::sha256_file(options.detector);
  const auto pose_hash = squatris::tools::sha256_file(options.pose);
  if (!detector_hash) {
    return detector_hash.error();
  }
  if (!pose_hash) {
    return pose_hash.error();
  }
  const auto equal_hash = [](std::string first, std::string second) {
    const auto lower = [](unsigned char value) { return static_cast<char>(std::tolower(value)); };
    std::ranges::transform(first, first.begin(), lower);
    std::ranges::transform(second, second.begin(), lower);
    return first == second;
  };
  if (!equal_hash(detector_hash.value(), *expected_detector) ||
      !equal_hash(pose_hash.value(), *expected_pose)) {
    return Status{ErrorCode::invalid_data,
                  "Model checksum does not match the authoritative manifest"};
  }
  return nlohmann::json{
      {"backend", "rtmpose-m-halpe26-256x192"},
      {"execution_provider",
       options.provider == "cuda" ? "CUDAExecutionProvider" : "CPUExecutionProvider"},
      {"detector_sha256", detector_hash.value()},
      {"pose_sha256", pose_hash.value()},
      {"checksums_verified", true},
      {"strict_provider_assignment", false},
      {"provider_evidence",
       {{"method", options.provider == "cuda" ? "onnxruntime-profiling" : "not-requested"},
        {"verified", options.provider != "cuda"},
        {"cuda_graph_execution", false},
        {"profiles", nlohmann::json::array()}}}};
}

Status validate_report(const nlohmann::json& report) {
  try {
    if (report.at("schema_version") != 3) {
      return {ErrorCode::invalid_data, "Unsupported benchmark report schema"};
    }
    const auto invalid = [](std::string message) {
      return Status{ErrorCode::invalid_data, std::move(message)};
    };
    const auto valid_hash = [](const nlohmann::json& value) {
      if (!value.is_string()) {
        return false;
      }
      const auto hash = value.get<std::string>();
      return hash.size() == 64 && std::ranges::all_of(hash, [](unsigned char character) {
               return std::isxdigit(character) != 0;
             });
    };
    const auto finite_nonnegative = [](double value) {
      return std::isfinite(value) && value >= 0.0;
    };
    const auto approximately_equal = [](double first, double second) {
      const auto scale = (std::max)(1.0, (std::max)(std::abs(first), std::abs(second)));
      return std::abs(first - second) <= scale * 1.0e-6;
    };
    const auto mode = report.at("configuration").at("mode").get<std::string>();
    if (mode != "deterministic" && mode != "realtime") {
      return {ErrorCode::invalid_data, "Benchmark report has an invalid mode"};
    }
    const auto provider = report.at("model").at("execution_provider").get<std::string>();
    if (provider != "deterministic-cpu" && provider != "CPUExecutionProvider" &&
        provider != "CUDAExecutionProvider") {
      return {ErrorCode::invalid_data, "Benchmark report has an unknown provider"};
    }
    const auto& model = report.at("model");
    for (const auto* key : {"detector_sha256", "pose_sha256"}) {
      if (!model.at(key).is_null() && !valid_hash(model.at(key))) {
        return invalid(std::string("Benchmark report has an invalid ") + key);
      }
    }
    const auto expected_warmup =
        report.at("configuration").at("warmup_iterations").get<std::size_t>();
    const auto expected_measured =
        report.at("configuration").at("measured_iterations").get<std::size_t>();
    const auto& warmup = report.at("warmup");
    const auto& measured = report.at("measured");
    std::optional<std::uint64_t> previous_sequence;
    std::optional<std::uint64_t> previous_source_index;
    std::optional<std::int64_t> previous_media_timestamp;
    std::uint64_t derived_skips{};
    const auto validate_phase = [&](const nlohmann::json& phase, std::size_t expected_max,
                                    std::string_view phase_name) -> Status {
      const auto frames = phase.at("frames_processed").get<std::size_t>();
      const auto& records = phase.at("processed_frames");
      if (!records.is_array() || records.size() != frames || frames > expected_max) {
        return invalid(std::string(phase_name) + " processed frame count is inconsistent");
      }
      std::uint64_t pose_available{};
      std::uint64_t usable_pose{};
      std::uint64_t full_body{};
      std::array<std::uint64_t, body_regions.size()> left{};
      std::array<std::uint64_t, body_regions.size()> right{};
      std::array<std::uint64_t, body_regions.size()> complete{};
      constexpr std::array timing_names{"frame_acquisition",
                                        "estimator_end_to_end",
                                        "frame_age_at_inference_start",
                                        "detector_total",
                                        "detector_preprocessing",
                                        "detector_inference",
                                        "detector_postprocessing",
                                        "pose_preprocessing",
                                        "pose_inference",
                                        "pose_postprocessing",
                                        "capture_delivery_to_pose_completion"};
      for (const auto& record : records) {
        const auto sequence = record.at("sequence").get<std::uint64_t>();
        if (sequence == 0 || (previous_sequence.has_value() && sequence <= *previous_sequence)) {
          return invalid("Benchmark frame sequences are not strictly increasing");
        }
        if (previous_sequence.has_value()) {
          const auto gap = sequence - *previous_sequence - 1;
          derived_skips += gap;
          if (mode == "deterministic" && gap != 0) {
            return invalid("Deterministic benchmark frame sequences are not contiguous");
          }
        }
        previous_sequence = sequence;
        const auto& source_index = record.at("source_frame_index");
        if (!source_index.is_null()) {
          const auto index = source_index.get<std::uint64_t>();
          if (previous_source_index.has_value() &&
              (index <= *previous_source_index ||
               (mode == "deterministic" && index != *previous_source_index + 1))) {
            return invalid("Source frame indices are out of order");
          }
          previous_source_index = index;
        }
        const auto& media_timestamp = record.at("source_media_timestamp_us");
        if (!media_timestamp.is_null()) {
          const auto timestamp = media_timestamp.get<std::int64_t>();
          if (timestamp < 0 ||
              (previous_media_timestamp.has_value() && timestamp < *previous_media_timestamp)) {
            return invalid("Source media timestamps are out of order");
          }
          previous_media_timestamp = timestamp;
        }
        const auto has_pose = record.at("pose_available").get<bool>();
        const auto usable = record.at("usable_pose").get<bool>();
        const auto is_full_body = record.at("full_body_complete").get<bool>();
        const auto tracking_state = record.at("tracking_state").get<std::string>();
        if (usable != (tracking_state == "usable-pose") || (usable && !has_pose) ||
            (is_full_body && !has_pose)) {
          return invalid("Tracking state and pose availability are inconsistent");
        }
        pose_available += has_pose ? 1 : 0;
        usable_pose += usable ? 1 : 0;
        full_body += is_full_body ? 1 : 0;
        const auto& coverage = record.at("landmark_coverage");
        bool derived_full_body = has_pose;
        for (std::size_t index = 0; index < body_regions.size(); ++index) {
          const auto& region = coverage.at(std::string(body_regions[index].name));
          const auto left_available = region.at("left").get<bool>();
          const auto right_available = region.at("right").get<bool>();
          const auto pair = region.at("complete").get<bool>();
          if (pair != (left_available && right_available)) {
            return invalid("Landmark coverage pair completeness is inconsistent");
          }
          left[index] += left_available ? 1 : 0;
          right[index] += right_available ? 1 : 0;
          complete[index] += pair ? 1 : 0;
          derived_full_body = derived_full_body && pair;
        }
        if (derived_full_body != is_full_body) {
          return invalid("Full-body completeness does not match landmark coverage");
        }
        const auto& frame_timings = record.at("timings_ms");
        for (const auto* timing : timing_names) {
          if (!finite_nonnegative(frame_timings.at(timing).get<double>())) {
            return invalid("Per-frame timing is invalid");
          }
        }
        const auto detector_parts = frame_timings.at("detector_preprocessing").get<double>() +
                                    frame_timings.at("detector_inference").get<double>() +
                                    frame_timings.at("detector_postprocessing").get<double>();
        if (frame_timings.at("detector_total").get<double>() + 0.1 < detector_parts) {
          return invalid("Detector timing components exceed detector total");
        }
        const auto capture_to_completion =
            frame_timings.at("frame_age_at_inference_start").get<double>() +
            frame_timings.at("estimator_end_to_end").get<double>();
        if (!approximately_equal(
                capture_to_completion,
                frame_timings.at("capture_delivery_to_pose_completion").get<double>())) {
          return invalid("Capture-to-completion timing is inconsistent");
        }
      }
      if (phase.at("no_pose_frames").get<std::uint64_t>() != frames - pose_available ||
          phase.at("pose_available_frames").get<std::uint64_t>() != pose_available ||
          phase.at("usable_pose_frames").get<std::uint64_t>() != usable_pose ||
          phase.at("full_body_complete_frames").get<std::uint64_t>() != full_body) {
        return invalid(std::string(phase_name) + " pose counts are inconsistent");
      }
      const auto& aggregate_coverage = phase.at("landmark_coverage");
      for (std::size_t index = 0; index < body_regions.size(); ++index) {
        const auto& region = aggregate_coverage.at(std::string(body_regions[index].name));
        if (region.at("left_available_frames").get<std::uint64_t>() != left[index] ||
            region.at("right_available_frames").get<std::uint64_t>() != right[index] ||
            region.at("complete_frames").get<std::uint64_t>() != complete[index]) {
          return invalid(std::string(phase_name) + " landmark coverage counts are inconsistent");
        }
      }
      const auto elapsed = phase.at("elapsed_seconds").get<double>();
      const auto throughput = phase.at("throughput_fps").get<double>();
      const auto useful_throughput = phase.at("useful_pose_throughput_fps").get<double>();
      if (!finite_nonnegative(elapsed) || !finite_nonnegative(throughput) ||
          !finite_nonnegative(useful_throughput) || (frames > 0 && elapsed <= 0.0) ||
          (elapsed > 0.0 &&
           (!approximately_equal(throughput, static_cast<double>(frames) / elapsed) ||
            !approximately_equal(useful_throughput, static_cast<double>(usable_pose) / elapsed)))) {
        return invalid(std::string(phase_name) + " throughput or elapsed time is inconsistent");
      }
      const auto& timings = phase.at("timings");
      for (const auto* timing : timing_names) {
        const auto& summary = timings.at(timing);
        const auto samples = summary.at("samples").get<std::size_t>();
        const auto median = summary.at("median_ms").get<double>();
        const auto p95 = summary.at("p95_ms").get<double>();
        const auto p99 = summary.at("p99_ms").get<double>();
        if (samples != frames || !finite_nonnegative(median) || !finite_nonnegative(p95) ||
            !finite_nonnegative(p99) || median > p95 || p95 > p99 ||
            (samples == 0 && (median != 0.0 || p95 != 0.0 || p99 != 0.0))) {
          return invalid(std::string(phase_name) + " timing summary is inconsistent");
        }
      }
      return Status::success();
    };
    const auto warmup_validation = validate_phase(warmup, expected_warmup, "Warm-up");
    if (!warmup_validation.ok()) {
      return warmup_validation;
    }
    const auto measured_validation = validate_phase(measured, expected_measured, "Measured");
    if (!measured_validation.ok()) {
      return measured_validation;
    }
    const auto& status = report.at("status");
    const auto success = status.at("success").get<bool>();
    const auto error_count = status.at("error_count").get<std::uint64_t>();
    const auto deadline_exceeded = status.at("deadline_exceeded").get<bool>();
    const auto& terminal_error = status.at("terminal_error");
    if (success) {
      if (error_count != 0 || deadline_exceeded || !terminal_error.is_null() ||
          warmup.at("frames_processed").get<std::size_t>() != expected_warmup ||
          measured.at("frames_processed").get<std::size_t>() != expected_measured ||
          !model.at("checksums_verified").get<bool>()) {
        return invalid("Successful benchmark report violates completion invariants");
      }
    } else {
      if (error_count == 0 || !terminal_error.is_object() ||
          terminal_error.at("code").get<std::string>().empty() ||
          terminal_error.at("stage").get<std::string>().empty() ||
          terminal_error.at("message").get<std::string>().empty()) {
        return invalid("Failed benchmark report lacks a terminal error");
      }
      if (deadline_exceeded && terminal_error.at("code") != "timeout") {
        return invalid("Deadline failure does not carry a timeout error");
      }
      if (measured.at("frames_processed").get<std::size_t>() > 0 &&
          warmup.at("frames_processed").get<std::size_t>() != expected_warmup) {
        return invalid("Measured samples exist before warm-up completed");
      }
    }
    const auto frames_skipped = report.at("results").at("frames_skipped").get<std::uint64_t>();
    if ((success && frames_skipped != derived_skips) ||
        (mode == "deterministic" && frames_skipped != 0)) {
      return invalid("Benchmark skipped-frame evidence is inconsistent");
    }
    if (previous_sequence.has_value() &&
        report.at("results").at("source_frames_delivered").get<std::uint64_t>() <
            *previous_sequence) {
      return invalid("Source delivery count precedes recorded frame sequences");
    }
    const auto& input = report.at("input");
    if (!input.at("source_sha256").is_null() && !valid_hash(input.at("source_sha256"))) {
      return invalid("Benchmark input hash is malformed");
    }
    if (success && input.at("kind") == "video" && !valid_hash(input.at("source_sha256"))) {
      return invalid("Successful video benchmark lacks a valid source hash");
    }
    const auto strict_assignment = model.at("strict_provider_assignment").get<bool>();
    const auto& provider_evidence = model.at("provider_evidence");
    if (provider_evidence.at("cuda_graph_execution").get<bool>()) {
      return invalid("Benchmark incorrectly claims CUDA graph execution");
    }
    if (provider == "CUDAExecutionProvider" && success) {
      if (!strict_assignment || provider_evidence.at("method") != "onnxruntime-profiling" ||
          !provider_evidence.at("verified").get<bool>() ||
          provider_evidence.at("profiles").size() != 2) {
        return invalid("Successful CUDA benchmark lacks strict profiling evidence");
      }
      std::array<bool, 2> sessions{};
      for (const auto& profile : provider_evidence.at("profiles")) {
        const auto session = profile.at("session").get<std::string>();
        if (session == "detector") {
          sessions[0] = true;
        } else if (session == "pose") {
          sessions[1] = true;
        } else {
          return invalid("Provider profile identifies an unknown session");
        }
        if (!valid_hash(profile.at("sha256")) ||
            profile.at("cuda_node_events").get<std::uint64_t>() == 0 ||
            profile.at("cpu_node_events").get<std::uint64_t>() != 0 ||
            profile.at("other_node_events").get<std::uint64_t>() != 0 ||
            !profile.at("verified").get<bool>()) {
          return invalid("CUDA provider profile contains unverified node assignment");
        }
      }
      if (!sessions[0] || !sessions[1]) {
        return invalid("CUDA provider profiles do not cover both model sessions");
      }
    } else if (provider != "CUDAExecutionProvider" && strict_assignment) {
      return invalid("Non-CUDA report claims strict CUDA provider assignment");
    }
    const auto& runtime = report.at("runtime");
    if (!runtime.at("initialization_ms").is_null() &&
        !finite_nonnegative(runtime.at("initialization_ms").get<double>())) {
      return invalid("Initialization timing is invalid");
    }
    if (!runtime.at("first_inference_ms").is_null() &&
        !finite_nonnegative(runtime.at("first_inference_ms").get<double>())) {
      return invalid("First-inference timing is invalid");
    }
  } catch (const nlohmann::json::exception& exception) {
    return {ErrorCode::invalid_data,
            "Benchmark report is missing required evidence: " + std::string(exception.what())};
  }
  return Status::success();
}

int validate_report_file(const std::filesystem::path& path) {
  const auto report = load_json(path);
  if (!report) {
    std::cerr << report.error().message << '\n';
    return 1;
  }
  const auto validation = validate_report(report.value());
  if (!validation.ok()) {
    std::cerr << validation.message << '\n';
    return 1;
  }
  std::cout << "Benchmark report is valid: " << path << '\n';
  return 0;
}

}

int run_benchmark(int argc, char** argv) {
  const auto options = parse(argc, argv);
  if (!options) {
    std::cerr << "Usage: SquatrisPoseBenchmark [--mode deterministic|realtime] [--video PATH] "
                 "[--backend mock|rtmpose] [--provider cpu|cuda] [--warmup N] "
                 "[--iterations N] [--deadline-seconds N] [--output PATH] or "
                 "--validate-report PATH\n";
    return 2;
  }
  if (!options->validate_report.empty()) {
    return validate_report_file(options->validate_report);
  }
  auto report = initial_report(*options);
  const auto persist = [&]() {
    const auto validation = validate_report(report);
    if (!validation.ok()) {
      std::cerr << "Generated report failed validation: " << validation.message << '\n';
      return false;
    }
    const auto written = write_report(options->output, report);
    if (!written.ok()) {
      std::cerr << written.message << '\n';
      return false;
    }
    return true;
  };
  const auto fail = [&](const Status& error, std::string_view stage, std::uint64_t errors = 1) {
    record_failure(report, error, stage, errors);
    persist();
    std::cerr << error.message << '\n';
    return 1;
  };
  if (!persist()) {
    return 1;
  }

  if (!options->video.empty()) {
    const auto video_hash = squatris::tools::sha256_file(options->video);
    if (!video_hash) {
      return fail(video_hash.error(), "input-hash");
    }
    report["input"]["source_sha256"] = video_hash.value();
  }
  const auto model_identity = verified_model_identity(*options);
  if (!model_identity) {
    return fail(model_identity.error(), "model-verification");
  }
  report["model"] = model_identity.value();
  if (!persist()) {
    return 1;
  }

  const auto initialization_started = squatris::pose::Clock::now();
  auto estimator = make_estimator(*options);
  report["runtime"]["initialization_ms"] =
      std::chrono::duration<double, std::milli>(squatris::pose::Clock::now() -
                                                initialization_started)
          .count();
  if (!estimator) {
    return fail(estimator.error(), "estimator-initialization");
  }
  if (estimator.value()->execution_provider() !=
      model_identity.value().at("execution_provider").get<std::string>()) {
    return fail(Status{ErrorCode::initialization_failed,
                       "Estimator provider does not match the requested provider"},
                "provider-selection");
  }
  auto source = make_source(*options);
  const auto source_status = source->start();
  if (!source_status.ok()) {
    return fail(source_status, "source-initialization");
  }

  PhaseStatistics warmup(options->warmup);
  PhaseStatistics measured(options->iterations);
  const squatris::inference::TrackingPolicy tracking_policy{};
  std::uint64_t sequence{};
  std::uint64_t skipped{};
  std::uint64_t errors{};
  std::size_t consecutive_failures{};
  std::size_t completed{};
  const auto total = options->warmup + options->iterations;
  const auto deadline =
      squatris::pose::Clock::now() + std::chrono::seconds(options->deadline_seconds);
  std::optional<Status> terminal_error;
  std::string terminal_stage;
  bool deadline_exceeded{};

  while (completed < total) {
    if (squatris::pose::Clock::now() >= deadline) {
      deadline_exceeded = true;
      terminal_error = Status{ErrorCode::timeout, "Benchmark execution deadline exceeded"};
      terminal_stage = "execution-deadline";
      break;
    }
    auto frame =
        source->wait_for_frame(sequence, std::chrono::milliseconds(options->frame_timeout_ms));
    if (!frame) {
      ++errors;
      ++consecutive_failures;
      if (frame.error().code != ErrorCode::timeout ||
          consecutive_failures >= options->maximum_consecutive_failures) {
        terminal_error = frame.error();
        terminal_stage = "frame-source";
        break;
      }
      continue;
    }
    if (sequence != 0 && frame.value()->sequence > sequence + 1) {
      skipped += frame.value()->sequence - sequence - 1;
    }
    sequence = frame.value()->sequence;
    auto& phase = completed < options->warmup ? warmup : measured;
    if (!phase.started_at) {
      phase.started_at = squatris::pose::Clock::now();
    }
    auto result = estimator.value()->infer(*frame.value());
    if (!result) {
      ++errors;
      ++consecutive_failures;
      std::cerr << "Inference error: " << result.error().message << '\n';
      if (consecutive_failures >= options->maximum_consecutive_failures) {
        terminal_error = Status{ErrorCode::inference_failed,
                                "Persistent inference failure: " + result.error().message};
        terminal_stage = "inference";
        break;
      }
      continue;
    }
    consecutive_failures = 0;
    if (report["runtime"]["first_inference_ms"].is_null()) {
      report["runtime"]["first_inference_ms"] = milliseconds(result.value().timings.end_to_end);
    }
    if (completed < options->warmup) {
      add_sample(warmup, *frame.value(), result.value(), tracking_policy);
    } else {
      add_sample(measured, *frame.value(), result.value(), tracking_policy);
    }
    ++completed;
  }
  source->stop();

  const auto format = source->format();
  report["input"]["width"] = format.width;
  report["input"]["height"] = format.height;
  report["input"]["source_fps"] = format.negotiated_fps;
  report["warmup"] = phase_json(warmup);
  report["measured"] = phase_json(measured);
  report["results"] = {{"frames_skipped", skipped}, {"source_frames_delivered", sequence}};

  if (options->backend == "rtmpose" && options->provider == "cuda") {
    const auto evidence = provider_profiling_evidence(*estimator.value());
    if (evidence) {
      report["model"]["provider_evidence"] = evidence.value();
      report["model"]["strict_provider_assignment"] = evidence.value().at("verified").get<bool>();
      if (!evidence.value().at("verified").get<bool>() && !terminal_error.has_value()) {
        terminal_error = Status{ErrorCode::initialization_failed,
                                "ONNX Runtime profiling did not prove exclusive CUDA assignment "
                                "for both model sessions"};
        terminal_stage = "provider-evidence";
        ++errors;
      }
    } else if (!terminal_error.has_value()) {
      terminal_error = evidence.error();
      terminal_stage = "provider-evidence";
      ++errors;
    }
  }
  if (errors > 0 && !terminal_error.has_value()) {
    terminal_error =
        Status{ErrorCode::inference_failed, "Benchmark encountered one or more transient errors"};
    terminal_stage = "execution";
  }
  const auto success = completed == total && errors == 0 && !terminal_error.has_value() &&
                       (!deadline_exceeded) &&
                       (options->mode != BenchmarkMode::deterministic || skipped == 0);
  if (success) {
    report["status"] = {{"success", true},
                        {"error_count", 0},
                        {"deadline_exceeded", false},
                        {"terminal_error", nullptr}};
  } else {
    if (!terminal_error.has_value()) {
      terminal_error = Status{ErrorCode::inference_failed,
                              "Benchmark ended without satisfying success invariants"};
      terminal_stage = "completion";
    }
    record_failure(report, *terminal_error,
                   terminal_stage.empty() ? std::string_view("execution")
                                          : std::string_view(terminal_stage),
                   errors);
  }
  if (!persist()) {
    return 1;
  }
  std::cout << report.dump(2) << '\n';
  return success ? 0 : 1;
}

int main(int argc, char** argv) noexcept {
  try {
    return run_benchmark(argc, argv);
  } catch (const std::exception& exception) {
    std::fputs("Benchmark failed with an unexpected error: ", stderr);
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
  } catch (...) {
    std::fputs("Benchmark failed with an unknown error\n", stderr);
  }
  return 1;
}
