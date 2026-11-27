#include "squatris/backends/rtmpose/rtmpose_estimator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "squatris/backends/rtmpose/model_math.hpp"
#include "squatris/core/pose/mapping.hpp"

#if SQUATRIS_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace squatris::backends::rtmpose {

#if SQUATRIS_HAS_ONNXRUNTIME

namespace {

constexpr int detector_width = 416;
constexpr int detector_height = 416;
constexpr int pose_width = 192;
constexpr int pose_height = 256;
constexpr std::size_t halpe_keypoints = 26;

struct BoundingBox {
  float left{};
  float top{};
  float right{};
  float bottom{};
  float score{};
};

struct ModelNames {
  std::vector<Ort::AllocatedStringPtr> storage;
  std::vector<const char*> values;
};

void require_contract(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("ONNX model contract mismatch: " + message);
  }
}

void validate_tensor_metadata(const Ort::Session& session, bool input, std::size_t index,
                              ONNXTensorElementDataType expected_type,
                              std::span<const std::int64_t> expected_shape,
                              std::string_view tensor_name) {
  const auto type_info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
  require_contract(type_info.GetONNXType() == ONNX_TYPE_TENSOR,
                   std::string(tensor_name) + " is not a tensor");
  const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  require_contract(tensor_info.GetElementType() == expected_type,
                   std::string(tensor_name) + " element type is incorrect");
  const auto shape = tensor_info.GetShape();
  require_contract(shape.size() == expected_shape.size(),
                   std::string(tensor_name) + " rank is incorrect");
  for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
    if (expected_shape[dimension] != 0) {
      require_contract(shape[dimension] == expected_shape[dimension],
                       std::string(tensor_name) + " dimension " + std::to_string(dimension) +
                           " is " + std::to_string(shape[dimension]) + ", expected " +
                           std::to_string(expected_shape[dimension]));
    }
  }
}

void validate_model_contracts(const Ort::Session& detector, const ModelNames& detector_inputs,
                              const ModelNames& detector_outputs, const Ort::Session& pose,
                              const ModelNames& pose_inputs, const ModelNames& pose_outputs) {
  require_contract(detector_inputs.values.size() == 1 &&
                       std::string_view(detector_inputs.values[0]) == "input",
                   "detector input name must be input");
  require_contract(detector_outputs.values.size() == 2 &&
                       std::string_view(detector_outputs.values[0]) == "dets" &&
                       std::string_view(detector_outputs.values[1]) == "labels",
                   "detector outputs must be dets and labels");
  require_contract(pose_inputs.values.size() == 1 &&
                       std::string_view(pose_inputs.values[0]) == "input",
                   "pose input name must be input");
  require_contract(pose_outputs.values.size() == 2 &&
                       std::string_view(pose_outputs.values[0]) == "simcc_x" &&
                       std::string_view(pose_outputs.values[1]) == "simcc_y",
                   "pose outputs must be simcc_x and simcc_y");
  constexpr std::array<std::int64_t, 4> detector_input_shape{1, 3, detector_height, detector_width};
  constexpr std::array<std::int64_t, 3> detector_output_shape{1, 0, 5};
  constexpr std::array<std::int64_t, 2> detector_label_shape{1, 0};
  constexpr std::array<std::int64_t, 4> pose_input_shape{0, 3, pose_height, pose_width};
  constexpr std::array<std::int64_t, 3> dynamic_pose_output_shape{0, 0, 0};
  validate_tensor_metadata(detector, true, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                           detector_input_shape, "detector input");
  validate_tensor_metadata(detector, false, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                           detector_output_shape, "detector dets");
  validate_tensor_metadata(detector, false, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                           detector_label_shape, "detector labels");
  validate_tensor_metadata(pose, true, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, pose_input_shape,
                           "pose input");
  validate_tensor_metadata(pose, false, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                           dynamic_pose_output_shape, "pose simcc_x");
  validate_tensor_metadata(pose, false, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                           dynamic_pose_output_shape, "pose simcc_y");
}

ModelNames input_names(const Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
  ModelNames names;
  const auto count = session.GetInputCount();
  names.storage.reserve(count);
  names.values.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    names.storage.push_back(session.GetInputNameAllocated(index, allocator));
    names.values.push_back(names.storage.back().get());
  }
  return names;
}

ModelNames output_names(const Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
  ModelNames names;
  const auto count = session.GetOutputCount();
  names.storage.reserve(count);
  names.values.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    names.storage.push_back(session.GetOutputNameAllocated(index, allocator));
    names.values.push_back(names.storage.back().get());
  }
  return names;
}

Ort::Env make_environment() {
  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "squatris");
  environment.DisableTelemetryEvents();
  return environment;
}

Ort::SessionOptions session_options(ExecutionProvider provider, int device_id) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  if (provider == ExecutionProvider::cuda) {
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
#if SQUATRIS_ORT_HAS_CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = device_id;
    options.AppendExecutionProvider_CUDA(cuda_options);
#else
    static_cast<void>(device_id);
    throw std::runtime_error("The selected ONNX Runtime package does not contain CUDA support");
#endif
  }
  return options;
}

bool provider_available(std::string_view expected) {
  const auto providers = Ort::GetAvailableProviders();
  return std::ranges::find(providers, expected) != providers.end();
}

std::vector<float> bgr_to_nchw(const cv::Mat& image) {
  std::vector<float> tensor(static_cast<std::size_t>(image.rows) * image.cols * 3);
  const auto plane = static_cast<std::size_t>(image.rows) * image.cols;
  for (int y = 0; y < image.rows; ++y) {
    const auto* row = image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < image.cols; ++x) {
      const auto offset = static_cast<std::size_t>(y) * image.cols + x;
      tensor[offset] = static_cast<float>(row[x][0]);
      tensor[plane + offset] = static_cast<float>(row[x][1]);
      tensor[2 * plane + offset] = static_cast<float>(row[x][2]);
    }
  }
  return tensor;
}

std::vector<float> normalized_rgb_to_nchw(const cv::Mat& bgr_image) {
  std::vector<float> tensor(static_cast<std::size_t>(bgr_image.rows) * bgr_image.cols * 3);
  const auto plane = static_cast<std::size_t>(bgr_image.rows) * bgr_image.cols;
  for (int y = 0; y < bgr_image.rows; ++y) {
    const auto* row = bgr_image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < bgr_image.cols; ++x) {
      const auto offset = static_cast<std::size_t>(y) * bgr_image.cols + x;
      const auto normalized = normalize_pose_pixel({row[x][0], row[x][1], row[x][2]});
      for (std::size_t channel = 0; channel < normalized.size(); ++channel) {
        tensor[channel * plane + offset] = normalized[channel];
      }
    }
  }
  return tensor;
}

cv::Point2f third_point(cv::Point2f first, cv::Point2f second) {
  const auto direction = first - second;
  return {second.x - direction.y, second.y + direction.x};
}

cv::Mat affine_pose_crop(const cv::Mat& source, const BoundingBox& box, cv::Point2f& center,
                         cv::Size2f& scale) {
  const auto region = prepare_pose_region({box.left, box.top, box.right, box.bottom});
  if (!region) {
    throw std::runtime_error(region.error().message);
  }
  center = {region.value().center_x, region.value().center_y};
  scale = {region.value().width, region.value().height};
  std::array<cv::Point2f, 3> source_points{
      center, center + cv::Point2f{0.0F, -scale.width * 0.5F}, {}};
  source_points[2] = third_point(source_points[0], source_points[1]);
  std::array<cv::Point2f, 3> destination_points{
      cv::Point2f{pose_width * 0.5F, pose_height * 0.5F},
      cv::Point2f{pose_width * 0.5F, pose_height * 0.5F - pose_width * 0.5F},
      {}};
  destination_points[2] = third_point(destination_points[0], destination_points[1]);
  const auto transform = cv::getAffineTransform(source_points.data(), destination_points.data());
  cv::Mat crop;
  cv::warpAffine(source, crop, transform, {pose_width, pose_height}, cv::INTER_LINEAR);
  return crop;
}

}

struct RtmposeEstimator::Impl {
  explicit Impl(RtmposeConfig source_config)
      : config(std::move(source_config)), environment(make_environment()),
        detector(environment, config.detector_model.c_str(),
                 session_options(config.provider, config.cuda_device_id)),
        pose(environment, config.pose_model.c_str(),
             session_options(config.provider, config.cuda_device_id)),
        detector_inputs(input_names(detector, allocator)),
        detector_outputs(output_names(detector, allocator)),
        pose_inputs(input_names(pose, allocator)), pose_outputs(output_names(pose, allocator)) {
    validate_model_contracts(detector, detector_inputs, detector_outputs, pose, pose_inputs,
                             pose_outputs);
  }

  RtmposeConfig config;
  Ort::Env environment;
  Ort::Session detector;
  Ort::Session pose;
  Ort::AllocatorWithDefaultOptions allocator;
  ModelNames detector_inputs;
  ModelNames detector_outputs;
  ModelNames pose_inputs;
  ModelNames pose_outputs;
};

#else

struct RtmposeEstimator::Impl {};

#endif

RtmposeEstimator::RtmposeEstimator(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

RtmposeEstimator::~RtmposeEstimator() = default;
RtmposeEstimator::RtmposeEstimator(RtmposeEstimator&&) noexcept = default;
RtmposeEstimator& RtmposeEstimator::operator=(RtmposeEstimator&&) noexcept = default;

Result<std::unique_ptr<RtmposeEstimator>> RtmposeEstimator::create(const RtmposeConfig& config) {
#if SQUATRIS_HAS_ONNXRUNTIME
  if (!std::filesystem::is_regular_file(config.detector_model) ||
      !std::filesystem::is_regular_file(config.pose_model)) {
    return Status{ErrorCode::not_found, "RTMPose detector or pose model was not found"};
  }
  if (!std::isfinite(config.detection_threshold) || config.detection_threshold < 0.0F ||
      config.detection_threshold > 1.0F || !std::isfinite(config.landmark_threshold) ||
      config.landmark_threshold < 0.0F || config.landmark_threshold > 1.0F) {
    return Status{ErrorCode::invalid_argument, "RTMPose thresholds must be in [0, 1]"};
  }
  const auto expected =
      config.provider == ExecutionProvider::cuda ? "CUDAExecutionProvider" : "CPUExecutionProvider";
  if (!provider_available(expected)) {
    return Status{ErrorCode::unavailable,
                  std::string("Requested ONNX Runtime provider is unavailable: ") + expected};
  }
  try {
    return std::unique_ptr<RtmposeEstimator>(new RtmposeEstimator(std::make_unique<Impl>(config)));
  } catch (const std::exception& exception) {
    return Status{ErrorCode::initialization_failed, exception.what()};
  }
#else
  static_cast<void>(config);
  return Status{ErrorCode::unavailable,
                "RTMPose backend was not compiled; configure with SQUATRIS_ENABLE_ONNXRUNTIME=ON"};
#endif
}

Result<squatris::inference::InferenceResult> RtmposeEstimator::infer(const input::Frame& frame) {
#if SQUATRIS_HAS_ONNXRUNTIME
  if (!implementation_) {
    return Status{ErrorCode::initialization_failed, "RTMPose backend is not initialized"};
  }
  if (frame.width == 0 || frame.height == 0 || frame.stride < frame.width * 3 ||
      frame.pixels.size() < static_cast<std::size_t>(frame.stride) * frame.height) {
    return Status{ErrorCode::invalid_data, "Input frame storage is invalid"};
  }
  const auto started = pose::Clock::now();
  squatris::inference::InferenceResult result;
  result.backend = backend_name();
  result.execution_provider = execution_provider();
  result.timings.frame_acquisition = frame.acquisition_duration;
  result.timings.frame_age_at_start =
      std::chrono::duration_cast<std::chrono::microseconds>(started - frame.captured_at);
  try {
    const cv::Mat source(static_cast<int>(frame.height), static_cast<int>(frame.width), CV_8UC3,
                         const_cast<std::uint8_t*>(frame.pixels.data()), frame.stride);
    const auto detection_started = pose::Clock::now();
    const auto ratio = std::min(static_cast<float>(detector_width) / frame.width,
                                static_cast<float>(detector_height) / frame.height);
    const cv::Size resized_size{static_cast<int>(static_cast<float>(frame.width) * ratio),
                                static_cast<int>(static_cast<float>(frame.height) * ratio)};
    cv::Mat resized;
    cv::resize(source, resized, resized_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat detector_image(detector_height, detector_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(detector_image(cv::Rect(0, 0, resized.cols, resized.rows)));
    auto detector_tensor_data = bgr_to_nchw(detector_image);
    constexpr std::array<std::int64_t, 4> detector_shape{1, 3, detector_height, detector_width};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto detector_tensor = Ort::Value::CreateTensor<float>(
        memory, detector_tensor_data.data(), detector_tensor_data.size(), detector_shape.data(),
        detector_shape.size());
    auto detector_outputs = implementation_->detector.Run(
        Ort::RunOptions{nullptr}, implementation_->detector_inputs.values.data(), &detector_tensor,
        1, implementation_->detector_outputs.values.data(),
        implementation_->detector_outputs.values.size());
    if (detector_outputs.size() < 2 || !detector_outputs[0].IsTensor() ||
        !detector_outputs[1].IsTensor()) {
      return Status{ErrorCode::inference_failed, "Detector outputs do not match the manifest"};
    }
    const auto detection_info = detector_outputs[0].GetTensorTypeAndShapeInfo();
    const auto detection_shape = detection_info.GetShape();
    const auto label_info = detector_outputs[1].GetTensorTypeAndShapeInfo();
    const auto label_shape = label_info.GetShape();
    if (detection_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        detection_shape.size() != 3 || detection_shape[0] != 1 || detection_shape[1] < 0 ||
        detection_shape[2] != 5 ||
        label_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        label_shape.size() != 2 || label_shape[0] != 1 || label_shape[1] != detection_shape[1]) {
      return Status{ErrorCode::inference_failed, "Detector tensor shapes or types are invalid"};
    }
    const auto detection_count = static_cast<std::size_t>(detection_shape[1]);
    if (detection_info.GetElementCount() != detection_count * 5 ||
        label_info.GetElementCount() != detection_count) {
      return Status{ErrorCode::inference_failed, "Detector tensor element counts are invalid"};
    }
    const auto* detections = detector_outputs[0].GetTensorData<float>();
    const auto* labels = detector_outputs[1].GetTensorData<std::int64_t>();
    std::optional<BoundingBox> selected;
    for (std::size_t index = 0; index < detection_count; ++index) {
      const auto label = labels[index];
      const auto* detection = detections + index * 5;
      if (label != 0 ||
          !std::ranges::all_of(std::span(detection, 5),
                               [](float value) { return std::isfinite(value); }) ||
          detection[4] < implementation_->config.detection_threshold) {
        continue;
      }
      if (!selected || detection[4] > selected->score) {
        selected = BoundingBox{
            std::clamp(detection[0] / ratio, 0.0F, static_cast<float>(frame.width - 1)),
            std::clamp(detection[1] / ratio, 0.0F, static_cast<float>(frame.height - 1)),
            std::clamp(detection[2] / ratio, 0.0F, static_cast<float>(frame.width - 1)),
            std::clamp(detection[3] / ratio, 0.0F, static_cast<float>(frame.height - 1)),
            detection[4]};
      }
    }
    result.timings.person_detection = std::chrono::duration_cast<std::chrono::microseconds>(
        pose::Clock::now() - detection_started);
    if (!selected || selected->right <= selected->left || selected->bottom <= selected->top) {
      result.timings.end_to_end =
          std::chrono::duration_cast<std::chrono::microseconds>(pose::Clock::now() - started);
      return result;
    }
    result.person_detected = true;

    const auto preprocessing_started = pose::Clock::now();
    cv::Point2f center;
    cv::Size2f scale;
    const auto crop = affine_pose_crop(source, *selected, center, scale);
    auto pose_tensor_data = normalized_rgb_to_nchw(crop);
    result.timings.pose_preprocessing = std::chrono::duration_cast<std::chrono::microseconds>(
        pose::Clock::now() - preprocessing_started);

    constexpr std::array<std::int64_t, 4> pose_shape{1, 3, pose_height, pose_width};
    auto pose_tensor =
        Ort::Value::CreateTensor<float>(memory, pose_tensor_data.data(), pose_tensor_data.size(),
                                        pose_shape.data(), pose_shape.size());
    const auto inference_started = pose::Clock::now();
    auto pose_outputs = implementation_->pose.Run(
        Ort::RunOptions{nullptr}, implementation_->pose_inputs.values.data(), &pose_tensor, 1,
        implementation_->pose_outputs.values.data(), implementation_->pose_outputs.values.size());
    result.timings.pose_inference = std::chrono::duration_cast<std::chrono::microseconds>(
        pose::Clock::now() - inference_started);
    const auto postprocessing_started = pose::Clock::now();
    if (pose_outputs.size() < 2 || !pose_outputs[0].IsTensor() || !pose_outputs[1].IsTensor()) {
      return Status{ErrorCode::inference_failed, "Pose outputs do not match the manifest"};
    }
    const auto x_shape = pose_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto y_shape = pose_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    const auto x_info = pose_outputs[0].GetTensorTypeAndShapeInfo();
    const auto y_info = pose_outputs[1].GetTensorTypeAndShapeInfo();
    if (x_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        y_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || x_shape.size() != 3 ||
        y_shape.size() != 3 || x_shape[0] != 1 || y_shape[0] != 1 ||
        x_shape[1] != halpe_keypoints || y_shape[1] != halpe_keypoints ||
        x_shape[2] != pose_width * 2 || y_shape[2] != pose_height * 2 ||
        x_info.GetElementCount() != halpe_keypoints * pose_width * 2 ||
        y_info.GetElementCount() != halpe_keypoints * pose_height * 2) {
      return Status{ErrorCode::inference_failed, "Pose output shapes do not match HALPE-26 SimCC"};
    }
    const auto* simcc_x = pose_outputs[0].GetTensorData<float>();
    const auto* simcc_y = pose_outputs[1].GetTensorData<float>();
    const auto model_landmarks = decode_simcc(
        std::span(simcc_x, halpe_keypoints * static_cast<std::size_t>(x_shape[2])),
        std::span(simcc_y, halpe_keypoints * static_cast<std::size_t>(y_shape[2])), halpe_keypoints,
        static_cast<std::size_t>(x_shape[2]), static_cast<std::size_t>(y_shape[2]),
        {center.x, center.y, scale.width, scale.height});
    if (!model_landmarks) {
      return model_landmarks.error();
    }
    const auto mapped = pose::map_landmarks(pose::LandmarkLayout::halpe26, model_landmarks.value(),
                                            {frame.width, frame.height});
    if (!mapped) {
      return mapped.error();
    }
    pose::PoseFrame pose_frame;
    pose_frame.sequence = frame.sequence;
    pose_frame.captured_at = frame.captured_at;
    pose_frame.inference_completed_at = pose::Clock::now();
    pose_frame.image_size = {frame.width, frame.height};
    pose_frame.landmarks = mapped.value();
    pose_frame.inference_backend = backend_name();
    float confidence_sum{};
    std::size_t confidence_count{};
    for (auto& landmark : pose_frame.landmarks) {
      if (landmark.confidence &&
          *landmark.confidence < implementation_->config.landmark_threshold) {
        landmark = {};
      }
      if (landmark.confidence) {
        confidence_sum += *landmark.confidence;
        ++confidence_count;
      }
    }
    if (confidence_count == 0) {
      result.timings.pose_postprocessing = std::chrono::duration_cast<std::chrono::microseconds>(
          pose::Clock::now() - postprocessing_started);
      result.timings.end_to_end =
          std::chrono::duration_cast<std::chrono::microseconds>(pose::Clock::now() - started);
      return result;
    }
    pose_frame.pose_confidence = confidence_sum / static_cast<float>(confidence_count);
    const auto validation = pose::validate_pose(pose_frame);
    if (!validation.ok()) {
      return validation;
    }
    result.pose = std::move(pose_frame);
    result.timings.pose_postprocessing = std::chrono::duration_cast<std::chrono::microseconds>(
        pose::Clock::now() - postprocessing_started);
    result.timings.end_to_end =
        std::chrono::duration_cast<std::chrono::microseconds>(pose::Clock::now() - started);
    return result;
  } catch (const Ort::Exception& exception) {
    return Status{ErrorCode::inference_failed, exception.what()};
  } catch (const cv::Exception& exception) {
    return Status{ErrorCode::invalid_data, exception.what()};
  } catch (const std::exception& exception) {
    return Status{ErrorCode::inference_failed, exception.what()};
  }
#else
  static_cast<void>(frame);
  return Status{ErrorCode::unavailable, "RTMPose backend is not compiled"};
#endif
}

std::string RtmposeEstimator::backend_name() const { return "rtmpose-m-halpe26-256x192"; }

std::string RtmposeEstimator::execution_provider() const {
#if SQUATRIS_HAS_ONNXRUNTIME
  if (implementation_) {
    return implementation_->config.provider == ExecutionProvider::cuda ? "CUDAExecutionProvider"
                                                                       : "CPUExecutionProvider";
  }
#endif
  return "unavailable";
}

}
