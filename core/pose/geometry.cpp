#include "squatris/core/pose/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace squatris::pose {

std::optional<float> joint_angle_degrees(const Landmark& first, const Landmark& vertex,
                                         const Landmark& third, ImageSize image_size,
                                         float minimum_confidence) {
  if (image_size.width == 0 || image_size.height == 0) {
    return std::nullopt;
  }
  const auto valid = [minimum_confidence](const Landmark& landmark) {
    return landmark.availability == LandmarkAvailability::available && landmark.image.has_value() &&
           landmark.confidence.has_value() && *landmark.confidence >= minimum_confidence;
  };
  if (!valid(first) || !valid(vertex) || !valid(third)) {
    return std::nullopt;
  }
  const auto first_image = first.image.value_or(NormalizedImagePoint{});
  const auto vertex_image = vertex.image.value_or(NormalizedImagePoint{});
  const auto third_image = third.image.value_or(NormalizedImagePoint{});
  const auto width = static_cast<float>(image_size.width);
  const auto height = static_cast<float>(image_size.height);
  const auto ax = (first_image.x - vertex_image.x) * width;
  const auto ay = (first_image.y - vertex_image.y) * height;
  const auto bx = (third_image.x - vertex_image.x) * width;
  const auto by = (third_image.y - vertex_image.y) * height;
  const auto a_length = std::hypot(ax, ay);
  const auto b_length = std::hypot(bx, by);
  if (a_length <= 1.0e-6F || b_length <= 1.0e-6F) {
    return std::nullopt;
  }
  const auto cosine = std::clamp((ax * bx + ay * by) / (a_length * b_length), -1.0F, 1.0F);
  return std::acos(cosine) * 180.0F / std::numbers::pi_v<float>;
}

}
