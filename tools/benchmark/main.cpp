#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/version.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include "sha256.hpp"
#include "squatris/backends/camera/prerecorded_source.hpp"
#include "squatris/backends/camera/synthetic_source.hpp"
#include "squatris/backends/inference/mock_pose_estimator.hpp"
#include "squatris/backends/rtmpose/rtmpose_estimator.hpp"
#include "squatris/core/metrics/latency_statistics.hpp"

namespace {

using squatris::ErrorCode;
using squatris::Status;

struct Options {
  std::filesystem::path video;
  std::filesystem::path output{"benchmark-result.json"};
  std::filesystem::path detector{"models/detector.onnx"};
  std::filesystem::path pose{"models/pose.onnx"};
  std::string backend{"mock"};
  std::string provider{"cpu"};
  std::size_t warmup{20};
  std::size_t iterations{200};
  std::uint32_t width{640};
  std::uint32_t height{480};
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
    } else if (argument == "--backend") {
      options.backend = *value;
    } else if (argument == "--provider") {
      options.provider = *value;
    } else if (argument != "--warmup" && argument != "--iterations" && argument != "--width" &&
               argument != "--height") {
      std::cerr << "Unknown argument: " << argument << '\n';
      return std::nullopt;
    }
  }
  if (options.iterations == 0) {
    std::cerr << "Measured iterations must be nonzero\n";
    return std::nullopt;
  }
  return options;
}

std::unique_ptr<squatris::input::IFrameSource> make_source(const Options& options) {
  if (!options.video.empty()) {
    return std::make_unique<squatris::backends::camera::PrerecordedSource>(
        squatris::backends::camera::PrerecordedSourceConfig{options.video, false, true});
  }
  return std::make_unique<squatris::backends::camera::SyntheticSource>(
      squatris::backends::camera::SyntheticSourceConfig{options.width, options.height, 240.0});
}

squatris::Result<std::unique_ptr<squatris::inference::IPoseEstimator>>
make_estimator(const Options& options) {
  if (options.backend == "mock") {
    return std::unique_ptr<squatris::inference::IPoseEstimator>(
        std::make_unique<squatris::backends::inference::MockPoseEstimator>());
  }
  if (options.backend != "rtmpose") {
    return Status{ErrorCode::invalid_argument, "Backend must be mock or rtmpose"};
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

nlohmann::json percentile_json(const squatris::metrics::Percentiles& percentiles) {
  return {{"samples", percentiles.samples},
          {"median_ms", percentiles.median_ms},
          {"p95_ms", percentiles.p95_ms},
          {"p99_ms", percentiles.p99_ms}};
}

}

int main(int argc, char** argv) {
  const auto options = parse(argc, argv);
  if (!options) {
    std::cerr << "Usage: SquatrisPoseBenchmark [--video PATH] [--backend mock|rtmpose] "
                 "[--provider cpu|cuda] [--warmup N] [--iterations N] [--output PATH]\n";
    return 2;
  }
  auto estimator = make_estimator(*options);
  if (!estimator) {
    std::cerr << "Estimator initialization failed: " << estimator.error().message << '\n';
    return 1;
  }
  auto source = make_source(*options);
  const auto source_status = source->start();
  if (!source_status.ok()) {
    std::cerr << "Source initialization failed: " << source_status.message << '\n';
    return 1;
  }
  squatris::metrics::LatencyStatistics end_to_end(options->iterations);
  squatris::metrics::LatencyStatistics acquisition(options->iterations);
  squatris::metrics::LatencyStatistics frame_age(options->iterations);
  squatris::metrics::LatencyStatistics detector(options->iterations);
  squatris::metrics::LatencyStatistics preprocessing(options->iterations);
  squatris::metrics::LatencyStatistics inference(options->iterations);
  squatris::metrics::LatencyStatistics postprocessing(options->iterations);
  std::uint64_t sequence{};
  std::uint64_t skipped{};
  std::uint64_t errors{};
  std::uint64_t no_pose{};
  std::optional<squatris::pose::TimePoint> measurement_started;
  const auto total = options->warmup + options->iterations;
  std::size_t completed{};
  while (completed < total) {
    auto frame = source->wait_for_frame(sequence, std::chrono::seconds(5));
    if (!frame) {
      ++errors;
      if (frame.error().code == ErrorCode::timeout) {
        continue;
      }
      std::cerr << "Frame source failed: " << frame.error().message << '\n';
      break;
    }
    if (sequence != 0 && frame.value()->sequence > sequence + 1) {
      skipped += frame.value()->sequence - sequence - 1;
    }
    sequence = frame.value()->sequence;
    if (completed == options->warmup && !measurement_started) {
      measurement_started = squatris::pose::Clock::now();
    }
    auto result = estimator.value()->infer(*frame.value());
    if (!result) {
      ++errors;
      std::cerr << "Inference error: " << result.error().message << '\n';
      continue;
    }
    if (!result.value().pose && completed >= options->warmup) {
      ++no_pose;
    }
    if (completed >= options->warmup) {
      end_to_end.add(result.value().timings.end_to_end);
      acquisition.add(result.value().timings.frame_acquisition);
      frame_age.add(result.value().timings.frame_age_at_start);
      detector.add(result.value().timings.person_detection);
      preprocessing.add(result.value().timings.pose_preprocessing);
      inference.add(result.value().timings.pose_inference);
      postprocessing.add(result.value().timings.pose_postprocessing);
    }
    ++completed;
  }
  const auto benchmark_completed = squatris::pose::Clock::now();
  const auto benchmark_elapsed =
      measurement_started
          ? std::chrono::duration<double>(benchmark_completed - *measurement_started).count()
          : 0.0;
  source->stop();
  const auto measured_frames = completed > options->warmup ? completed - options->warmup : 0;
  nlohmann::json model_identity = {{"backend", estimator.value()->backend_name()},
                                   {"execution_provider", estimator.value()->execution_provider()}};
  if (options->backend == "rtmpose") {
    const auto detector_hash = squatris::tools::sha256_file(options->detector);
    const auto pose_hash = squatris::tools::sha256_file(options->pose);
    model_identity["detector_sha256"] = detector_hash ? detector_hash.value() : "unavailable";
    model_identity["pose_sha256"] = pose_hash ? pose_hash.value() : "unavailable";
  } else {
    model_identity["detector_sha256"] = nullptr;
    model_identity["pose_sha256"] = nullptr;
  }
  const auto format = source->format();
  nlohmann::json report = {
      {"schema_version", 1},
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
      {"model", model_identity},
      {"input",
       {{"source", options->video.empty() ? "synthetic" : options->video.string()},
        {"width", format.width},
        {"height", format.height},
        {"source_fps", format.negotiated_fps}}},
      {"configuration",
       {{"warmup_iterations", options->warmup}, {"measured_iterations", options->iterations}}},
      {"results",
       {{"frames_processed", measured_frames},
        {"frames_skipped", skipped},
        {"no_pose_frames", no_pose},
        {"error_count", errors},
        {"elapsed_seconds", benchmark_elapsed},
        {"throughput_fps",
         benchmark_elapsed > 0.0 ? static_cast<double>(measured_frames) / benchmark_elapsed : 0.0},
        {"frame_acquisition", percentile_json(acquisition.calculate())},
        {"end_to_end", percentile_json(end_to_end.calculate())},
        {"frame_age_at_inference_start", percentile_json(frame_age.calculate())},
        {"person_detection", percentile_json(detector.calculate())},
        {"pose_preprocessing", percentile_json(preprocessing.calculate())},
        {"pose_inference", percentile_json(inference.calculate())},
        {"pose_postprocessing", percentile_json(postprocessing.calculate())}}}};
  std::ofstream output(options->output);
  if (!output) {
    std::cerr << "Could not write " << options->output << '\n';
    return 1;
  }
  output << report.dump(2) << '\n';
  std::cout << report.dump(2) << '\n';
  return measured_frames == options->iterations && errors == 0 ? 0 : 1;
}
