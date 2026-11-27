#include "parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "report.hpp"

namespace squatris::benchmark {

int compare_parity_reports(const ParityOptions& options) {
  const auto cpu = load_json(options.compare_cpu_report);
  const auto cuda = load_json(options.compare_cuda_report);
  if (!cpu || !cuda) {
    std::cerr << (cpu ? cuda.error().message : cpu.error().message) << '\n';
    return 1;
  }
  const auto cpu_validation = validate_report(cpu.value());
  const auto cuda_validation = validate_report(cuda.value());
  if (!cpu_validation.ok() || !cuda_validation.ok()) {
    std::cerr << (cpu_validation.ok() ? cuda_validation.message : cpu_validation.message) << '\n';
    return 1;
  }
  constexpr double minimum_bbox_iou = 0.995;
  constexpr double maximum_coordinate_difference = 0.002;
  constexpr double maximum_confidence_difference = 0.005;
  constexpr double maximum_knee_angle_difference = 2.0;
  nlohmann::json errors = nlohmann::json::array();
  const auto& cpu_model = cpu.value().at("model");
  const auto& cuda_model = cuda.value().at("model");
  if (!cpu.value().at("status").at("success").get<bool>() ||
      !cuda.value().at("status").at("success").get<bool>() ||
      cpu_model.at("execution_policy") != "cpu" ||
      cuda_model.at("execution_policy") != "cuda-audited" ||
      cpu_model.at("detector_sha256") != cuda_model.at("detector_sha256") ||
      cpu_model.at("pose_sha256") != cuda_model.at("pose_sha256") ||
      cpu.value().at("input").at("source_sha256") != cuda.value().at("input").at("source_sha256") ||
      cpu.value().at("configuration").at("mode") != "deterministic" ||
      cuda.value().at("configuration").at("mode") != "deterministic") {
    errors.push_back("Reports do not describe successful comparable CPU and cuda-audited runs");
  }
  const auto& cpu_frames = cpu.value().at("measured").at("processed_frames");
  const auto& cuda_frames = cuda.value().at("measured").at("processed_frames");
  if (cpu_frames.size() != cuda_frames.size() || cpu_frames.empty()) {
    errors.push_back("Parity reports do not contain the same nonzero measured frame count");
  }
  std::size_t detection_mismatches{};
  std::size_t pose_mismatches{};
  std::size_t landmark_availability_mismatches{};
  std::size_t full_body_coverage_mismatches{};
  double minimum_observed_iou{1.0};
  double maximum_observed_coordinate_difference{};
  double maximum_observed_confidence_difference{};
  double maximum_observed_knee_angle_difference{};
  const auto count = (std::min)(cpu_frames.size(), cuda_frames.size());
  for (std::size_t index = 0; index < count; ++index) {
    const auto& first = cpu_frames[index];
    const auto& second = cuda_frames[index];
    if (first.at("source_frame_index") != second.at("source_frame_index") ||
        first.at("source_media_timestamp_us") != second.at("source_media_timestamp_us")) {
      errors.push_back("CPU and CUDA source frame identities differ at measured index " +
                       std::to_string(index));
      continue;
    }
    const auto first_detected = first.at("person_detected").get<bool>();
    const auto second_detected = second.at("person_detected").get<bool>();
    detection_mismatches += first_detected != second_detected ? 1 : 0;
    if (first_detected && second_detected) {
      const auto& first_box = first.at("selected_person");
      const auto& second_box = second.at("selected_person");
      const auto intersection_width = (std::max)(
          0.0,
          (std::min)(first_box.at("right").get<double>(), second_box.at("right").get<double>()) -
              (std::max)(first_box.at("left").get<double>(), second_box.at("left").get<double>()));
      const auto intersection_height = (std::max)(
          0.0,
          (std::min)(first_box.at("bottom").get<double>(), second_box.at("bottom").get<double>()) -
              (std::max)(first_box.at("top").get<double>(), second_box.at("top").get<double>()));
      const auto intersection = intersection_width * intersection_height;
      const auto first_area =
          (first_box.at("right").get<double>() - first_box.at("left").get<double>()) *
          (first_box.at("bottom").get<double>() - first_box.at("top").get<double>());
      const auto second_area =
          (second_box.at("right").get<double>() - second_box.at("left").get<double>()) *
          (second_box.at("bottom").get<double>() - second_box.at("top").get<double>());
      const auto iou = intersection / (first_area + second_area - intersection);
      minimum_observed_iou = (std::min)(minimum_observed_iou, iou);
      maximum_observed_confidence_difference =
          (std::max)(maximum_observed_confidence_difference,
                     std::abs(first_box.at("confidence").get<double>() -
                              second_box.at("confidence").get<double>()));
    }
    const auto first_pose = first.at("pose_available").get<bool>();
    const auto second_pose = second.at("pose_available").get<bool>();
    pose_mismatches += first_pose != second_pose ? 1 : 0;
    if (first_pose && second_pose) {
      maximum_observed_confidence_difference =
          (std::max)(maximum_observed_confidence_difference,
                     std::abs(first.at("pose_confidence").get<double>() -
                              second.at("pose_confidence").get<double>()));
    }
    full_body_coverage_mismatches +=
        first.at("full_body_complete") != second.at("full_body_complete") ? 1 : 0;
    const auto& first_landmarks = first.at("landmarks");
    const auto& second_landmarks = second.at("landmarks");
    for (std::size_t landmark_index = 0; landmark_index < first_landmarks.size();
         ++landmark_index) {
      const auto& first_landmark = first_landmarks[landmark_index];
      const auto& second_landmark = second_landmarks[landmark_index];
      const auto first_available = first_landmark.at("available").get<bool>();
      const auto second_available = second_landmark.at("available").get<bool>();
      landmark_availability_mismatches += first_available != second_available ? 1 : 0;
      if (first_available && second_available) {
        maximum_observed_coordinate_difference = (std::max)(
            {maximum_observed_coordinate_difference,
             std::abs(first_landmark.at("x").get<double>() - second_landmark.at("x").get<double>()),
             std::abs(first_landmark.at("y").get<double>() -
                      second_landmark.at("y").get<double>())});
        maximum_observed_confidence_difference =
            (std::max)(maximum_observed_confidence_difference,
                       std::abs(first_landmark.at("confidence").get<double>() -
                                second_landmark.at("confidence").get<double>()));
      }
    }
    for (const auto* side : {"left", "right"}) {
      const auto& first_angle = first.at("projected_knee_angles_degrees").at(side);
      const auto& second_angle = second.at("projected_knee_angles_degrees").at(side);
      if (first_angle.is_number() != second_angle.is_number()) {
        ++landmark_availability_mismatches;
      } else if (first_angle.is_number()) {
        maximum_observed_knee_angle_difference =
            (std::max)(maximum_observed_knee_angle_difference,
                       std::abs(first_angle.get<double>() - second_angle.get<double>()));
      }
    }
  }
  if (detection_mismatches != 0 || pose_mismatches != 0 || landmark_availability_mismatches != 0 ||
      full_body_coverage_mismatches != 0) {
    errors.push_back("CPU and audited CUDA availability or coverage differs");
  }
  if (minimum_observed_iou < minimum_bbox_iou) {
    errors.push_back("CPU and audited CUDA bounding-box IoU is below tolerance");
  }
  if (maximum_observed_coordinate_difference > maximum_coordinate_difference) {
    errors.push_back("CPU and audited CUDA landmark coordinate difference exceeds tolerance");
  }
  if (maximum_observed_confidence_difference > maximum_confidence_difference) {
    errors.push_back("CPU and audited CUDA confidence difference exceeds tolerance");
  }
  if (maximum_observed_knee_angle_difference > maximum_knee_angle_difference) {
    errors.push_back("CPU and audited CUDA projected knee-angle difference exceeds tolerance");
  }
  const nlohmann::json parity = {
      {"schema_version", 1},
      {"status", {{"success", errors.empty()}, {"errors", errors}}},
      {"cpu_report", std::filesystem::absolute(options.compare_cpu_report).string()},
      {"cuda_report", std::filesystem::absolute(options.compare_cuda_report).string()},
      {"model",
       {{"detector_sha256", cpu_model.at("detector_sha256")},
        {"pose_sha256", cpu_model.at("pose_sha256")}}},
      {"source_sha256", cpu.value().at("input").at("source_sha256")},
      {"frames_compared", count},
      {"tolerances",
       {{"minimum_bounding_box_iou", minimum_bbox_iou},
        {"maximum_normalized_landmark_coordinate_difference", maximum_coordinate_difference},
        {"maximum_confidence_difference", maximum_confidence_difference},
        {"maximum_projected_knee_angle_difference_degrees", maximum_knee_angle_difference}}},
      {"observed",
       {{"detection_availability_mismatches", detection_mismatches},
        {"pose_availability_mismatches", pose_mismatches},
        {"landmark_availability_mismatches", landmark_availability_mismatches},
        {"full_body_coverage_mismatches", full_body_coverage_mismatches},
        {"minimum_bounding_box_iou", minimum_observed_iou},
        {"maximum_normalized_landmark_coordinate_difference",
         maximum_observed_coordinate_difference},
        {"maximum_confidence_difference", maximum_observed_confidence_difference},
        {"maximum_projected_knee_angle_difference_degrees",
         maximum_observed_knee_angle_difference}}},
      {"anatomical_mapping", "canonical subject-left and subject-right identities compared"},
      {"physical_accuracy_claim", false}};
  const auto written = write_report(options.parity_output, parity);
  if (!written.ok()) {
    std::cerr << written.message << '\n';
    return 1;
  }
  std::cout << parity.dump(2) << '\n';
  return errors.empty() ? 0 : 1;
}

}
