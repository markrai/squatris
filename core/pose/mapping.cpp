#include "squatris/core/pose/mapping.hpp"

#include <array>
#include <utility>

namespace squatris::pose {

namespace {

using Mapping = std::pair<std::size_t, LandmarkId>;

constexpr std::array coco17{
    Mapping{0, LandmarkId::nose},           Mapping{5, LandmarkId::left_shoulder},
    Mapping{6, LandmarkId::right_shoulder}, Mapping{7, LandmarkId::left_elbow},
    Mapping{8, LandmarkId::right_elbow},    Mapping{9, LandmarkId::left_wrist},
    Mapping{10, LandmarkId::right_wrist},   Mapping{11, LandmarkId::left_hip},
    Mapping{12, LandmarkId::right_hip},     Mapping{13, LandmarkId::left_knee},
    Mapping{14, LandmarkId::right_knee},    Mapping{15, LandmarkId::left_ankle},
    Mapping{16, LandmarkId::right_ankle}};

constexpr std::array halpe26{Mapping{0, LandmarkId::nose},
                             Mapping{5, LandmarkId::left_shoulder},
                             Mapping{6, LandmarkId::right_shoulder},
                             Mapping{7, LandmarkId::left_elbow},
                             Mapping{8, LandmarkId::right_elbow},
                             Mapping{9, LandmarkId::left_wrist},
                             Mapping{10, LandmarkId::right_wrist},
                             Mapping{11, LandmarkId::left_hip},
                             Mapping{12, LandmarkId::right_hip},
                             Mapping{13, LandmarkId::left_knee},
                             Mapping{14, LandmarkId::right_knee},
                             Mapping{15, LandmarkId::left_ankle},
                             Mapping{16, LandmarkId::right_ankle},
                             Mapping{24, LandmarkId::left_heel},
                             Mapping{25, LandmarkId::right_heel},
                             Mapping{20, LandmarkId::left_foot_index},
                             Mapping{21, LandmarkId::right_foot_index}};

template <std::size_t Size>
Result<std::array<Landmark, landmark_count>> apply_mapping(const std::array<Mapping, Size>& mapping,
                                                           std::span<const ModelLandmark> source,
                                                           ImageSize image_size) {
  std::array<Landmark, landmark_count> result{};
  for (const auto& [source_index, target_id] : mapping) {
    if (source_index >= source.size()) {
      return Status{ErrorCode::invalid_data, "Model landmark count does not match its layout"};
    }
    const auto& value = source[source_index];
    const auto point = normalize_image_point(value.pixel_x, value.pixel_y, image_size);
    if (!point || value.confidence < 0.0F || value.confidence > 1.0F) {
      continue;
    }
    result[static_cast<std::size_t>(target_id)] =
        make_landmark(point.value(), std::nullopt, value.confidence);
  }
  return result;
}

}

Result<std::array<Landmark, landmark_count>>
map_landmarks(LandmarkLayout layout, std::span<const ModelLandmark> source, ImageSize image_size) {
  if (layout == LandmarkLayout::coco17) {
    return apply_mapping(coco17, source, image_size);
  }
  return apply_mapping(halpe26, source, image_size);
}

}
