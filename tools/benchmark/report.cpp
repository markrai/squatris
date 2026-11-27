#include "report.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "squatris/backends/rtmpose/provider_audit.hpp"
#include "squatris/core/crypto/sha256.hpp"
#include "squatris/core/io/atomic_file.hpp"
#include "squatris/core/pose/pose.hpp"

#include "body_regions.hpp"

namespace squatris::benchmark {

using squatris::ErrorCode;
using squatris::Status;
using squatris::benchmark::body_regions;

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

Status write_report(const std::filesystem::path& path, const nlohmann::json& report) {
  try {
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }
    return squatris::io::write_text_atomically(path, report.dump(2) + '\n');
  } catch (const std::exception& exception) {
    return {ErrorCode::io_error,
            "Could not write " + path.string() + ": " + std::string(exception.what())};
  }
}

Status validate_strict_profile_reference(const nlohmann::json& expected) {
  const auto path = expected.at("path").get<std::filesystem::path>();
  const auto hash = squatris::crypto::sha256_file(path);
  if (!hash || hash.value() != expected.at("sha256").get<std::string>()) {
    return {ErrorCode::invalid_data, "Strict CUDA profile hash does not match its artifact"};
  }
  const auto trace = load_json(path);
  if (!trace || !trace.value().is_array()) {
    return trace ? Status{ErrorCode::invalid_data, "Strict CUDA profile is not an event array"}
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
  if (cuda_nodes != expected.at("cuda_node_events").get<std::uint64_t>() ||
      cpu_nodes != expected.at("cpu_node_events").get<std::uint64_t>() ||
      other_nodes != expected.at("other_node_events").get<std::uint64_t>()) {
    return {ErrorCode::invalid_data, "Strict CUDA profile contents do not match the report"};
  }
  return Status::success();
}

Status validate_audited_profile_evidence(const nlohmann::json& evidence,
                                         const std::string& detector_hash,
                                         const std::string& pose_hash,
                                         const std::string& runtime_version) {
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
  try {
    if (evidence.at("method") != "onnxruntime-profile-plus-pinned-policy" ||
        !evidence.at("verified").get<bool>() ||
        !valid_hash(evidence.at("policy_manifest_sha256")) || evidence.at("sessions").size() != 2 ||
        evidence.at("runtime_version") != runtime_version ||
        evidence.at("runtime_header_version") != runtime_version ||
        evidence.at("static_assignment_complete").get<bool>() ||
        !evidence.at("runtime_artifacts").is_array() ||
        evidence.at("runtime_artifacts").size() != 3) {
      return invalid("Audited CUDA evidence lacks a verified pinned policy");
    }
    std::set<std::string> runtime_modules;
    for (const auto& artifact : evidence.at("runtime_artifacts")) {
      const auto name = artifact.at("name").get<std::string>();
      const auto path = artifact.at("path").get<std::filesystem::path>();
      const auto size = artifact.at("size_bytes").get<std::uintmax_t>();
      const auto hash = squatris::crypto::sha256_file(path);
      std::error_code size_error;
      const auto actual_size = std::filesystem::file_size(path, size_error);
      if (path.filename() != name || !path.is_absolute() || size == 0 ||
          !runtime_modules.insert(name).second || !valid_hash(artifact.at("sha256")) || !hash ||
          hash.value() != artifact.at("sha256").get<std::string>() || size_error ||
          actual_size != size) {
        return invalid("Loaded ONNX Runtime artifact does not match its bound evidence");
      }
    }
    if (!runtime_modules.contains("onnxruntime.dll") ||
        !runtime_modules.contains("onnxruntime_providers_shared.dll") ||
        !runtime_modules.contains("onnxruntime_providers_cuda.dll")) {
      return invalid("Audited CUDA evidence does not bind every required runtime module");
    }
    std::array<bool, 2> sessions{};
    std::vector<squatris::backends::rtmpose::ProviderProfile> profiles;
    for (const auto& session : evidence.at("sessions")) {
      const auto name = session.at("session").get<std::string>();
      if (name == "detector") {
        sessions[0] = true;
      } else if (name == "pose") {
        sessions[1] = true;
      } else {
        return invalid("Audited provider evidence identifies an unknown session");
      }
      if (!valid_hash(session.at("sha256")) || session.at("cuda_nodes").empty() ||
          !session.at("unexpected_cpu_nodes").empty() ||
          !session.at("unknown_provider_nodes").empty() || !session.at("verified").get<bool>()) {
        return invalid("Audited CUDA provider evidence violates the pinned policy");
      }
      profiles.push_back({name, session.at("path").get<std::filesystem::path>()});
    }
    if (!sessions[0] || !sessions[1]) {
      return invalid("Audited provider evidence does not cover both model sessions");
    }
    const auto reconstructed = squatris::backends::rtmpose::analyze_provider_profiles(
        profiles, evidence.at("policy_manifest").get<std::filesystem::path>(), detector_hash,
        pose_hash, evidence.at("runtime_version").get<std::string>(),
        evidence.at("runtime_header_version").get<std::string>(),
        evidence.at("runtime_artifacts"), evidence.at("verified_sessions_retained").get<bool>());
    if (!reconstructed || reconstructed.value() != evidence) {
      return invalid("Audited provider artifacts do not reproduce the reported evidence");
    }
    return Status::success();
  } catch (const nlohmann::json::exception& exception) {
    return invalid("Audited provider evidence is malformed: " + std::string(exception.what()));
  }
}

Status validate_report(const nlohmann::json& report) {
  try {
    if (report.at("schema_version") != 4) {
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
    const auto& executable = report.at("build").at("executable");
    if (!executable.is_object() || executable.at("path").get<std::string>().empty() ||
        !valid_hash(executable.at("sha256"))) {
      return invalid("Benchmark report lacks a verified executable identity");
    }
    const auto approximately_equal = [](double first, double second) {
      const auto scale = (std::max)(1.0, (std::max)(std::abs(first), std::abs(second)));
      return std::abs(first - second) <= scale * 1.0e-6;
    };
    const auto mode = report.at("configuration").at("mode").get<std::string>();
    if (mode != "deterministic" && mode != "realtime") {
      return {ErrorCode::invalid_data, "Benchmark report has an invalid mode"};
    }
    const auto& configuration = report.at("configuration");
    const auto& input_metadata = report.at("input");
    const auto& simulated_rate = configuration.at("simulated_source_fps");
    const auto duration_mode = !configuration.at("duration_seconds").is_null();
    if (!simulated_rate.is_null()) {
      const auto rate = simulated_rate.get<double>();
      if (!std::isfinite(rate) || rate <= 0.0 || rate > 240.0 || mode != "realtime" ||
          input_metadata.at("kind") != "video" || input_metadata.at("pacing") != "simulated-rate") {
        return invalid("Simulated source rate evidence is inconsistent");
      }
    } else if (input_metadata.at("pacing") != "native-media-timestamps") {
      return invalid("Source pacing evidence is inconsistent");
    }
    const auto loop_video = report.at("configuration").at("loop_video").get<bool>();
    if ((duration_mode && mode != "realtime") ||
        ((duration_mode || loop_video) && input_metadata.at("kind") != "video")) {
      return invalid("Duration and loop evidence require a valid video mode");
    }
    const auto& model = report.at("model");
    const auto provider = model.at("execution_provider").get<std::string>();
    if (provider != "deterministic-cpu" && provider != "CPUExecutionProvider" &&
        provider != "CUDAExecutionProvider" &&
        provider != "CUDAExecutionProvider+CPUExecutionProvider") {
      return {ErrorCode::invalid_data, "Benchmark report has an unknown provider"};
    }
    const auto policy = model.at("execution_policy").get<std::string>();
    if ((policy == "cpu" && provider != "deterministic-cpu" &&
         provider != "CPUExecutionProvider") ||
        (policy == "cuda-strict" && provider != "CUDAExecutionProvider") ||
        (policy == "cuda-audited" && provider != "CUDAExecutionProvider+CPUExecutionProvider") ||
        (policy != "cpu" && policy != "cuda-strict" && policy != "cuda-audited")) {
      return {ErrorCode::invalid_data, "Benchmark execution policy and provider are inconsistent"};
    }
    for (const auto* key : {"detector_sha256", "pose_sha256"}) {
      if (!model.at(key).is_null() && !valid_hash(model.at(key))) {
        return invalid(std::string("Benchmark report has an invalid ") + key);
      }
    }
    const auto expected_warmup =
        report.at("configuration").at("warmup_iterations").get<std::size_t>();
    const auto expected_measured =
        duration_mode ? (std::numeric_limits<std::size_t>::max)()
                      : report.at("configuration").at("measured_iterations").get<std::size_t>();
    const auto& warmup = report.at("warmup");
    const auto& measured = report.at("measured");
    std::optional<std::uint64_t> previous_sequence;
    std::optional<std::uint64_t> previous_source_index;
    std::optional<std::uint64_t> previous_loop_index;
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
      std::map<std::string, std::vector<double>> timing_samples;
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
        const auto& loop_index_value = record.at("source_loop_index");
        if (!loop_index_value.is_null()) {
          const auto loop_index = loop_index_value.get<std::uint64_t>();
          if (!loop_video ||
              (previous_loop_index.has_value() &&
               (loop_index < *previous_loop_index || loop_index > *previous_loop_index + 1))) {
            return invalid("Source loop identities are inconsistent");
          }
          if (previous_loop_index.has_value() && loop_index != *previous_loop_index) {
            previous_source_index.reset();
            previous_media_timestamp.reset();
          }
          previous_loop_index = loop_index;
        } else if (loop_video) {
          return invalid("Looped input is missing a loop identity");
        }
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
        const auto person_detected = record.at("person_detected").get<bool>();
        const auto& selected_person = record.at("selected_person");
        if (person_detected != selected_person.is_object()) {
          return invalid("Person detection and selected bounding box are inconsistent");
        }
        if (selected_person.is_object()) {
          const auto box_left = selected_person.at("left").get<double>();
          const auto top = selected_person.at("top").get<double>();
          const auto box_right = selected_person.at("right").get<double>();
          const auto bottom = selected_person.at("bottom").get<double>();
          const auto confidence = selected_person.at("confidence").get<double>();
          if (!std::isfinite(box_left) || !std::isfinite(top) || !std::isfinite(box_right) ||
              !std::isfinite(bottom) || !std::isfinite(confidence) || box_left < 0.0 || top < 0.0 ||
              box_right > 1.0 || bottom > 1.0 || box_left >= box_right || top >= bottom ||
              confidence < 0.0 || confidence > 1.0) {
            return invalid("Selected person bounding box is invalid");
          }
        }
        const auto& landmarks = record.at("landmarks");
        if (!landmarks.is_array() || landmarks.size() != squatris::pose::landmark_count) {
          return invalid("Canonical landmark evidence has an invalid size");
        }
        const auto& pose_confidence = record.at("pose_confidence");
        if (has_pose != pose_confidence.is_number()) {
          return invalid("Pose availability and confidence are inconsistent");
        }
        if (pose_confidence.is_number()) {
          const auto confidence = pose_confidence.get<double>();
          if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0) {
            return invalid("Pose confidence is invalid");
          }
        }
        for (std::size_t index = 0; index < landmarks.size(); ++index) {
          const auto& landmark = landmarks[index];
          const auto id = static_cast<squatris::pose::LandmarkId>(index);
          const auto available = landmark.at("available").get<bool>();
          if (landmark.at("name") != squatris::pose::to_string(id) || (!has_pose && available)) {
            return invalid("Canonical landmark identity or availability is inconsistent");
          }
          if (available) {
            const auto x = landmark.at("x").get<double>();
            const auto y = landmark.at("y").get<double>();
            const auto confidence = landmark.at("confidence").get<double>();
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(confidence) || x < 0.0 ||
                x > 1.0 || y < 0.0 || y > 1.0 || confidence < 0.0 || confidence > 1.0) {
              return invalid("Canonical landmark value is invalid");
            }
          } else if (!landmark.at("x").is_null() || !landmark.at("y").is_null() ||
                     !landmark.at("confidence").is_null()) {
            return invalid("Unavailable canonical landmark contains values");
          }
        }
        const auto& knee_angles = record.at("projected_knee_angles_degrees");
        for (const auto* side : {"left", "right"}) {
          if (!knee_angles.at(side).is_null()) {
            const auto angle = knee_angles.at(side).get<double>();
            if (!std::isfinite(angle) || angle < 0.0 || angle > 180.0) {
              return invalid("Projected knee angle is invalid");
            }
          }
        }
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
          const auto sample = frame_timings.at(timing).get<double>();
          if (!finite_nonnegative(sample)) {
            return invalid("Per-frame timing is invalid");
          }
          timing_samples[timing].push_back(sample);
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
        auto recomputed = timing_samples[timing];
        std::ranges::sort(recomputed);
        const auto percentile = [&recomputed](double probability) {
          if (recomputed.empty()) {
            return 0.0;
          }
          const auto position = probability * static_cast<double>(recomputed.size() - 1);
          const auto lower = static_cast<std::size_t>(std::floor(position));
          const auto upper = static_cast<std::size_t>(std::ceil(position));
          const auto fraction = position - static_cast<double>(lower);
          return recomputed[lower] * (1.0 - fraction) + recomputed[upper] * fraction;
        };
        if (samples != frames || !finite_nonnegative(median) || !finite_nonnegative(p95) ||
            !finite_nonnegative(p99) || median > p95 || p95 > p99 ||
            (samples == 0 && (median != 0.0 || p95 != 0.0 || p99 != 0.0)) ||
            !approximately_equal(median, percentile(0.5)) ||
            !approximately_equal(p95, percentile(0.95)) ||
            !approximately_equal(p99, percentile(0.99))) {
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
          (!duration_mode &&
           measured.at("frames_processed").get<std::size_t>() != expected_measured) ||
          (duration_mode && measured.at("elapsed_seconds").get<double>() + 1.0e-6 <
                                report.at("configuration").at("duration_seconds").get<double>()) ||
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
    if (success) {
      const auto native_rate = input.at("native_media_fps").get<double>();
      const auto delivery_rate = input.at("delivery_target_fps").get<double>();
      const auto source_rate = input.at("source_fps").get<double>();
      if (!std::isfinite(native_rate) || native_rate <= 0.0 ||
          !std::isfinite(delivery_rate) || delivery_rate <= 0.0 ||
          !approximately_equal(delivery_rate, source_rate) ||
          !approximately_equal(delivery_rate, simulated_rate.is_null()
                                                  ? native_rate
                                                  : simulated_rate.get<double>())) {
        return invalid("Source rate metadata is inconsistent");
      }
    }
    if (!input.at("source_sha256").is_null() && !valid_hash(input.at("source_sha256"))) {
      return invalid("Benchmark input hash is malformed");
    }
    if (success && input.at("kind") == "video" && !valid_hash(input.at("source_sha256"))) {
      return invalid("Successful video benchmark lacks a valid source hash");
    }
    const auto profiling_enabled = configuration.at("production_profiling_enabled").get<bool>();
    const auto recreate_sessions =
        configuration.at("recreate_audited_sessions_for_experiment").get<bool>();
    if ((profiling_enabled || recreate_sessions) && policy != "cuda-audited") {
      return invalid("CUDA diagnostic configuration requires cuda-audited");
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
        const auto artifact_validation = validate_strict_profile_reference(profile);
        if (!artifact_validation.ok()) {
          return artifact_validation;
        }
      }
      if (!sessions[0] || !sessions[1]) {
        return invalid("CUDA provider profiles do not cover both model sessions");
      }
    } else if (provider == "CUDAExecutionProvider+CPUExecutionProvider" && success) {
      if (strict_assignment) {
        return invalid("Audited CUDA benchmark claims strict provider assignment");
      }
      if (provider_evidence.at("verified_sessions_retained").get<bool>() !=
          (!profiling_enabled && !recreate_sessions)) {
        return invalid("Audited session retention does not match diagnostic configuration");
      }
      const auto artifact_validation = validate_audited_profile_evidence(
          provider_evidence, model.at("detector_sha256").get<std::string>(),
          model.at("pose_sha256").get<std::string>(),
          report.at("build").at("onnxruntime").get<std::string>());
      if (!artifact_validation.ok()) {
        return artifact_validation;
      }
    } else if (provider != "CUDAExecutionProvider" && strict_assignment) {
      return invalid("Non-CUDA report claims strict CUDA provider assignment");
    }
    const auto& runtime = report.at("runtime");
    const auto& production_profiling = runtime.at("production_profiling");
    if (production_profiling.at("enabled").get<bool>() != profiling_enabled) {
      return invalid("Production profiling configuration is inconsistent");
    }
    if (profiling_enabled && success) {
      if (policy != "cuda-audited" || !production_profiling.at("provider_evidence").is_object()) {
        return invalid("Profiled production run lacks audited CUDA evidence");
      }
      if (production_profiling.at("provider_evidence").at("verified_sessions_retained").get<bool>()) {
        return invalid("Diagnostic production profile incorrectly claims session retention");
      }
      const auto production_validation = validate_audited_profile_evidence(
          production_profiling.at("provider_evidence"),
          model.at("detector_sha256").get<std::string>(),
          model.at("pose_sha256").get<std::string>(),
          report.at("build").at("onnxruntime").get<std::string>());
      if (!production_validation.ok()) {
        return production_validation;
      }
    } else if (!profiling_enabled && !production_profiling.at("provider_evidence").is_null()) {
      return invalid("Unprofiled production run contains profiling evidence");
    }
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
