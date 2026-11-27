#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
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
        preprocessing(capacity), inference(capacity), postprocessing(capacity) {
    frame_records.reserve(capacity);
  }

  squatris::metrics::LatencyStatistics end_to_end;
  squatris::metrics::LatencyStatistics acquisition;
  squatris::metrics::LatencyStatistics frame_age;
  squatris::metrics::LatencyStatistics detector;
  squatris::metrics::LatencyStatistics preprocessing;
  squatris::metrics::LatencyStatistics inference;
  squatris::metrics::LatencyStatistics postprocessing;
  std::vector<nlohmann::json> frame_records;
  std::optional<squatris::pose::TimePoint> started_at;
  squatris::pose::TimePoint completed_at{};
  std::uint64_t no_pose{};
};

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

void add_sample(PhaseStatistics& phase, const squatris::input::Frame& frame,
                const squatris::inference::InferenceResult& result) {
  phase.end_to_end.add(result.timings.end_to_end);
  phase.acquisition.add(result.timings.frame_acquisition);
  phase.frame_age.add(result.timings.frame_age_at_start);
  phase.detector.add(result.timings.person_detection);
  phase.preprocessing.add(result.timings.pose_preprocessing);
  phase.inference.add(result.timings.pose_inference);
  phase.postprocessing.add(result.timings.pose_postprocessing);
  if (!result.pose) {
    ++phase.no_pose;
  }
  nlohmann::json record = {{"sequence", frame.sequence},
                           {"person_detected", result.person_detected},
                           {"usable_pose", result.pose.has_value()}};
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
  return {{"frames_processed", processed},
          {"no_pose_frames", phase.no_pose},
          {"elapsed_seconds", elapsed},
          {"throughput_fps", elapsed > 0.0 ? static_cast<double>(processed) / elapsed : 0.0},
          {"processed_frames", phase.frame_records},
          {"timings",
           {{"frame_acquisition", percentile_json(phase.acquisition.calculate())},
            {"end_to_end", percentile_json(phase.end_to_end.calculate())},
            {"frame_age_at_inference_start", percentile_json(phase.frame_age.calculate())},
            {"person_detection", percentile_json(phase.detector.calculate())},
            {"pose_preprocessing", percentile_json(phase.preprocessing.calculate())},
            {"pose_inference", percentile_json(phase.inference.calculate())},
            {"pose_postprocessing", percentile_json(phase.postprocessing.calculate())}}}};
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

squatris::Result<nlohmann::json> verified_model_identity(const Options& options) {
  if (options.backend != "rtmpose") {
    return nlohmann::json{{"backend", "mock-pose-v1"},  {"execution_provider", "deterministic-cpu"},
                          {"detector_sha256", nullptr}, {"pose_sha256", nullptr},
                          {"checksums_verified", true}, {"strict_provider_assignment", false}};
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
  return nlohmann::json{{"backend", "rtmpose-m-halpe26-256x192"},
                        {"execution_provider", options.provider == "cuda" ? "CUDAExecutionProvider"
                                                                          : "CPUExecutionProvider"},
                        {"detector_sha256", detector_hash.value()},
                        {"pose_sha256", pose_hash.value()},
                        {"checksums_verified", true},
                        {"strict_provider_assignment", options.provider == "cuda"}};
}

Status validate_report(const nlohmann::json& report) {
  try {
    if (report.at("schema_version") != 2) {
      return {ErrorCode::invalid_data, "Unsupported benchmark report schema"};
    }
    const auto mode = report.at("configuration").at("mode").get<std::string>();
    if (mode != "deterministic" && mode != "realtime") {
      return {ErrorCode::invalid_data, "Benchmark report has an invalid mode"};
    }
    const auto provider = report.at("model").at("execution_provider").get<std::string>();
    if (provider != "deterministic-cpu" && provider != "CPUExecutionProvider" &&
        provider != "CUDAExecutionProvider") {
      return {ErrorCode::invalid_data, "Benchmark report has an unknown provider"};
    }
    if (!report.at("model").at("checksums_verified").get<bool>()) {
      return {ErrorCode::invalid_data, "Benchmark report did not verify model checksums"};
    }
    const auto expected_warmup =
        report.at("configuration").at("warmup_iterations").get<std::size_t>();
    const auto expected_measured =
        report.at("configuration").at("measured_iterations").get<std::size_t>();
    const auto& warmup = report.at("warmup");
    const auto& measured = report.at("measured");
    if (warmup.at("frames_processed").get<std::size_t>() != warmup.at("processed_frames").size() ||
        measured.at("frames_processed").get<std::size_t>() !=
            measured.at("processed_frames").size()) {
      return {ErrorCode::invalid_data, "Processed frame evidence count is inconsistent"};
    }
    if (report.at("status").at("success").get<bool>() &&
        (warmup.at("frames_processed").get<std::size_t>() != expected_warmup ||
         measured.at("frames_processed").get<std::size_t>() != expected_measured ||
         report.at("status").at("error_count").get<std::size_t>() != 0)) {
      return {ErrorCode::invalid_data, "Successful benchmark report has incomplete samples"};
    }
    if (mode == "deterministic" && report.at("results").at("frames_skipped") != 0) {
      return {ErrorCode::invalid_data, "Deterministic benchmark reported skipped frames"};
    }
    if (report.at("input").at("kind") == "video" &&
        report.at("input").at("source_sha256").is_null()) {
      return {ErrorCode::invalid_data, "Video benchmark lacks its source hash"};
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

  const auto model_identity = verified_model_identity(*options);
  if (!model_identity) {
    std::cerr << "Model verification failed: " << model_identity.error().message << '\n';
    return 1;
  }
  auto estimator = make_estimator(*options);
  if (!estimator) {
    std::cerr << "Estimator initialization failed: " << estimator.error().message << '\n';
    return 1;
  }
  if (estimator.value()->execution_provider() !=
      model_identity.value().at("execution_provider").get<std::string>()) {
    std::cerr << "Estimator provider does not match the requested provider\n";
    return 1;
  }
  auto source = make_source(*options);
  const auto source_status = source->start();
  if (!source_status.ok()) {
    std::cerr << "Source initialization failed: " << source_status.message << '\n';
    return 1;
  }

  PhaseStatistics warmup(options->warmup);
  PhaseStatistics measured(options->iterations);
  std::uint64_t sequence{};
  std::uint64_t skipped{};
  std::uint64_t errors{};
  std::size_t consecutive_failures{};
  std::size_t completed{};
  const auto total = options->warmup + options->iterations;
  const auto deadline =
      squatris::pose::Clock::now() + std::chrono::seconds(options->deadline_seconds);
  std::optional<Status> terminal_error;
  bool deadline_exceeded{};

  while (completed < total) {
    if (squatris::pose::Clock::now() >= deadline) {
      deadline_exceeded = true;
      terminal_error = Status{ErrorCode::timeout, "Benchmark execution deadline exceeded"};
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
        break;
      }
      continue;
    }
    consecutive_failures = 0;
    if (completed < options->warmup) {
      add_sample(warmup, *frame.value(), result.value());
    } else {
      add_sample(measured, *frame.value(), result.value());
    }
    ++completed;
  }
  source->stop();

  const auto format = source->format();
  nlohmann::json input_identity = {{"kind", options->video.empty() ? "synthetic" : "video"},
                                   {"source", options->video.empty()
                                                  ? "deterministic-generated-pattern"
                                                  : options->video.string()},
                                   {"source_sha256", nullptr},
                                   {"width", format.width},
                                   {"height", format.height},
                                   {"source_fps", format.negotiated_fps}};
  if (!options->video.empty()) {
    const auto video_hash = squatris::tools::sha256_file(options->video);
    if (video_hash) {
      input_identity["source_sha256"] = video_hash.value();
    } else if (!terminal_error) {
      terminal_error = video_hash.error();
    }
  }
  const auto success = completed == total && errors == 0 && !terminal_error.has_value() &&
                       (!deadline_exceeded) &&
                       (options->mode != BenchmarkMode::deterministic || skipped == 0);
  nlohmann::json report = {
      {"schema_version", 2},
      {"status",
       {{"success", success},
        {"error_count", errors},
        {"deadline_exceeded", deadline_exceeded},
        {"terminal_error", terminal_error
                               ? nlohmann::json{{"code", error_code_name(terminal_error->code)},
                                                {"message", terminal_error->message}}
                               : nlohmann::json(nullptr)}}},
      {"hardware",
       {{"processor", environment_value("PROCESSOR_IDENTIFIER")},
        {"logical_processors", std::thread::hardware_concurrency()},
        {"physical_memory_bytes", physical_memory_bytes()},
        {"display_adapters", display_adapters()},
        {"computer_name", environment_value("COMPUTERNAME")},
        {"os", environment_value("OS")}}},
      {"build",
       {{"configuration",
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
        {"onnxruntime_compiled", static_cast<bool>(SQUATRIS_HAS_ONNXRUNTIME)}}},
      {"model", model_identity.value()},
      {"input", input_identity},
      {"configuration",
       {{"mode", options->mode == BenchmarkMode::deterministic ? "deterministic" : "realtime"},
        {"warmup_iterations", options->warmup},
        {"measured_iterations", options->iterations},
        {"deadline_seconds", options->deadline_seconds},
        {"frame_timeout_ms", options->frame_timeout_ms},
        {"maximum_consecutive_failures", options->maximum_consecutive_failures}}},
      {"warmup", phase_json(warmup)},
      {"measured", phase_json(measured)},
      {"results", {{"frames_skipped", skipped}, {"source_frames_delivered", sequence}}}};

  const auto report_validation = validate_report(report);
  if (!report_validation.ok()) {
    std::cerr << "Generated report failed validation: " << report_validation.message << '\n';
    return 1;
  }
  std::ofstream output(options->output);
  if (!output) {
    std::cerr << "Could not write " << options->output << '\n';
    return 1;
  }
  output << report.dump(2) << '\n';
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
