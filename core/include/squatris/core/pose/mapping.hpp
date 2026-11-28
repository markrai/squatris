#pragma once

#include <span>

#include "squatris/core/pose/pose.hpp"

namespace squatris::pose {

enum class LandmarkLayout { coco17, halpe26, coco_wholebody133 };

struct ModelLandmark {
  float pixel_x{};
  float pixel_y{};
  float confidence{};
};

[[nodiscard]] Result<std::array<Landmark, landmark_count>>
map_landmarks(LandmarkLayout layout, std::span<const ModelLandmark> source, ImageSize image_size);

}
