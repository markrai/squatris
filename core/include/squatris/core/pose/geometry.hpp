#pragma once

#include <optional>

#include "squatris/core/pose/pose.hpp"

namespace squatris::pose {

[[nodiscard]] std::optional<float> joint_angle_degrees(const Landmark& first,
                                                       const Landmark& vertex,
                                                       const Landmark& third, ImageSize image_size,
                                                       float minimum_confidence);

}
