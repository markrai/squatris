#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "squatris/backends/rtmpose/rtmpose_estimator.hpp"
#include "squatris/core/crypto/sha256.hpp"
#include "squatris/core/io/atomic_file.hpp"
#include "squatris/core/metrics/latency_statistics.hpp"

namespace {

struct Options {
  std::filesystem::path detector{"models/detector.onnx"};
  std::filesystem::path pose{"models/pose.onnx"};
  std::filesystem::path policy{"config/rtmpose-cuda-policy.json"};
  std::filesystem::path output{"model-benchmark-result.json"};
  std::size_t warmup{100};
  std::size_t iterations{2000};
};

std::optional<std::string> next_value(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    return std::nullopt;
  }
  return argv[++index];
}

std::optional<Options> parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto value = next_value(index, argc, argv);
    if (!value) {
      return std::nullopt;
    }
    if (argument == "--detector") {
      options.detector = *value;
    } else if (argument == "--pose") {
      options.pose = *value;
    } else if (argument == "--policy-manifest") {
      options.policy = *value;
    } else if (argument == "--output") {
      options.output = *value;
    } else if (argument == "--warmup" || argument == "--iterations") {
      std::size_t parsed{};
      const auto result = std::from_chars(value->data(), value->data() + value->size(), parsed);
      if (result.ec != std::errc{} || result.ptr != value->data() + value->size() || parsed == 0) {
        return std::nullopt;
      }
      (argument == "--warmup" ? options.warmup : options.iterations) = parsed;
    } else {
      return std::nullopt;
    }
  }
  return options;
}

nlohmann::json percentiles(const squatris::metrics::LatencyStatistics& statistics) {
  const auto values = statistics.calculate();
  return {{"median_ms", values.median_ms},
          {"p95_ms", values.p95_ms},
          {"p99_ms", values.p99_ms},
          {"samples", values.samples}};
}

int run(int argc, char** argv) {
  const auto options = parse(argc, argv);
  if (!options) {
    std::cerr << "Usage: SquatrisModelBenchmark [--detector PATH] [--pose PATH] "
                 "[--policy-manifest PATH] [--warmup N] [--iterations N] [--output PATH]\n";
    return 2;
  }
  if (!options->output.parent_path().empty()) {
    std::filesystem::create_directories(options->output.parent_path());
  }
  squatris::backends::rtmpose::RtmposeConfig config;
  config.detector_model = options->detector;
  config.pose_model = options->pose;
  config.provider = squatris::backends::rtmpose::ExecutionProvider::cuda_audited;
  config.provider_policy_manifest = options->policy;
  config.profiling_output_prefix =
      options->output.parent_path() / (options->output.stem().string() + "-ort-profile");
  const auto initialization_started = std::chrono::steady_clock::now();
  auto estimator = squatris::backends::rtmpose::RtmposeEstimator::create(config);
  const auto initialization_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - initialization_started)
                                     .count();
  if (!estimator) {
    std::cerr << estimator.error().message << '\n';
    return 1;
  }
  for (std::size_t index = 0; index < options->warmup; ++index) {
    const auto result = estimator.value()->infer_synthetic_tensors();
    if (!result) {
      std::cerr << result.error().message << '\n';
      return 1;
    }
  }
  squatris::metrics::LatencyStatistics detector(options->iterations);
  squatris::metrics::LatencyStatistics pose(options->iterations);
  squatris::metrics::LatencyStatistics total(options->iterations);
  nlohmann::json samples = nlohmann::json::array();
  const auto measured_started = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < options->iterations; ++index) {
    const auto result = estimator.value()->infer_synthetic_tensors();
    if (!result) {
      std::cerr << result.error().message << '\n';
      return 1;
    }
    detector.add(result.value().detector_inference);
    pose.add(result.value().pose_inference);
    total.add(result.value().total);
    using Milliseconds = std::chrono::duration<double, std::milli>;
    samples.push_back({{"detector_ms", Milliseconds(result.value().detector_inference).count()},
                       {"pose_ms", Milliseconds(result.value().pose_inference).count()},
                       {"total_ms", Milliseconds(result.value().total).count()}});
  }
  const auto measured_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - measured_started).count();
  const auto evidence = estimator.value()->provider_policy_evidence();
  const auto detector_hash = squatris::crypto::sha256_file(options->detector);
  const auto pose_hash = squatris::crypto::sha256_file(options->pose);
  const auto executable = std::filesystem::absolute(argv[0]);
  const auto executable_hash = squatris::crypto::sha256_file(executable);
  if (!evidence || !detector_hash || !pose_hash || !executable_hash) {
    std::cerr << "Could not bind synthetic tensor evidence to its artifacts\n";
    return 1;
  }
  const nlohmann::json report = {
      {"schema_version", 1},
      {"status", {{"success", true}}},
      {"workload", "zero-filled-synthetic-input-tensors"},
      {"included_work", "synchronous detector and pose Ort::Session::Run with output retrieval"},
      {"excluded_work", "video decode, image preprocessing, detector postprocessing, pose postprocessing"},
      {"configuration", {{"warmup", options->warmup}, {"iterations", options->iterations}}},
      {"artifacts",
       {{"executable", executable.string()},
        {"executable_sha256", executable_hash.value()},
        {"detector_sha256", detector_hash.value()},
        {"pose_sha256", pose_hash.value()}}},
      {"runtime", {{"initialization_ms", initialization_ms}, {"provider_evidence", evidence.value()}}},
      {"results",
       {{"elapsed_seconds", measured_seconds},
        {"throughput_iterations_per_second",
         measured_seconds > 0.0 ? static_cast<double>(options->iterations) / measured_seconds
                                : 0.0},
        {"detector_inference", percentiles(detector)},
        {"pose_inference", percentiles(pose)},
        {"combined_model_execution", percentiles(total)},
        {"samples", samples}}}};
  const auto written = squatris::io::write_text_atomically(options->output, report.dump(2) + '\n');
  if (!written.ok()) {
    std::cerr << written.message << '\n';
    return 1;
  }
  std::cout << report.dump(2) << '\n';
  return 0;
}

}

int main(int argc, char** argv) noexcept {
  try {
    return run(argc, argv);
  } catch (const std::exception& exception) {
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
  }
}
