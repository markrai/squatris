#include "squatris/core/pose/pose.hpp"

#include <cmath>
#include <limits>

namespace squatris::pose {

namespace {

bool normalized(float value) { return std::isfinite(value) && value >= 0.0F && value <= 1.0F; }

}

const Landmark& PoseFrame::at(LandmarkId id) const noexcept {
  return landmarks[static_cast<std::size_t>(id)];
}

Landmark& PoseFrame::at(LandmarkId id) noexcept { return landmarks[static_cast<std::size_t>(id)]; }

Result<NormalizedImagePoint> normalize_image_point(float pixel_x, float pixel_y,
                                                   ImageSize image_size) {
  if (image_size.width == 0 || image_size.height == 0) {
    return Status{ErrorCode::invalid_argument, "Image dimensions must be nonzero"};
  }
  if (!std::isfinite(pixel_x) || !std::isfinite(pixel_y)) {
    return Status{ErrorCode::invalid_data, "Landmark coordinates must be finite"};
  }
  const auto x = pixel_x / static_cast<float>(image_size.width);
  const auto y = pixel_y / static_cast<float>(image_size.height);
  if (!normalized(x) || !normalized(y)) {
    return Status{ErrorCode::invalid_data, "Landmark coordinates are outside the image"};
  }
  return NormalizedImagePoint{x, y};
}

Landmark make_landmark(NormalizedImagePoint point, std::optional<float> depth, float confidence) {
  if (!normalized(point.x) || !normalized(point.y) || !std::isfinite(confidence) ||
      confidence < 0.0F || confidence > 1.0F || (depth.has_value() && !std::isfinite(*depth))) {
    return {};
  }
  Landmark landmark;
  landmark.availability = LandmarkAvailability::available;
  landmark.image = point;
  if (depth.has_value()) {
    landmark.estimated_depth = EstimatedDepth{*depth};
  }
  landmark.confidence = confidence;
  return landmark;
}

Status validate_pose(const PoseFrame& frame) {
  if (frame.image_size.width == 0 || frame.image_size.height == 0) {
    return {ErrorCode::invalid_data, "Pose image dimensions must be nonzero"};
  }
  if (frame.inference_completed_at < frame.captured_at) {
    return {ErrorCode::invalid_data, "Inference completion precedes capture"};
  }
  if (frame.pose_confidence.has_value() &&
      (!std::isfinite(*frame.pose_confidence) || *frame.pose_confidence < 0.0F ||
       *frame.pose_confidence > 1.0F)) {
    return {ErrorCode::invalid_data, "Pose confidence is invalid"};
  }
  for (const auto& landmark : frame.landmarks) {
    if (landmark.availability == LandmarkAvailability::unavailable) {
      if (landmark.image.has_value() || landmark.confidence.has_value() ||
          landmark.estimated_depth.has_value()) {
        return {ErrorCode::invalid_data, "Unavailable landmark contains data"};
      }
      continue;
    }
    if (!landmark.image.has_value() || !landmark.confidence.has_value()) {
      return {ErrorCode::invalid_data, "Available landmark lacks required data"};
    }
    if (!normalized(landmark.image->x) || !normalized(landmark.image->y) ||
        !normalized(*landmark.confidence)) {
      return {ErrorCode::invalid_data, "Available landmark contains invalid data"};
    }
    if (landmark.estimated_depth.has_value() && !std::isfinite(landmark.estimated_depth->value)) {
      return {ErrorCode::invalid_data, "Estimated depth is invalid"};
    }
  }
  if (frame.inference_backend.empty()) {
    return {ErrorCode::invalid_data, "Inference backend identifier is empty"};
  }
  return Status::success();
}

const char* to_string(LandmarkId id) noexcept {
  constexpr std::array names{"nose",
                             "left_shoulder",
                             "right_shoulder",
                             "left_elbow",
                             "right_elbow",
                             "left_wrist",
                             "right_wrist",
                             "left_hip",
                             "right_hip",
                             "left_knee",
                             "right_knee",
                             "left_ankle",
                             "right_ankle",
                             "left_heel",
                             "right_heel",
                             "left_foot_index",
                             "right_foot_index"};
  const auto index = static_cast<std::size_t>(id);
  return index < names.size() ? names[index] : "unknown";
}

}
