#pragma once

#include <array>
#include <string_view>

#include "squatris/core/pose/pose.hpp"

namespace squatris::benchmark {

struct BodyRegion {
  std::string_view name;
  pose::LandmarkId left;
  pose::LandmarkId right;
};

inline constexpr std::array body_regions{
    BodyRegion{"shoulders", pose::LandmarkId::left_shoulder, pose::LandmarkId::right_shoulder},
    BodyRegion{"elbows", pose::LandmarkId::left_elbow, pose::LandmarkId::right_elbow},
    BodyRegion{"wrists", pose::LandmarkId::left_wrist, pose::LandmarkId::right_wrist},
    BodyRegion{"hips", pose::LandmarkId::left_hip, pose::LandmarkId::right_hip},
    BodyRegion{"knees", pose::LandmarkId::left_knee, pose::LandmarkId::right_knee},
    BodyRegion{"ankles", pose::LandmarkId::left_ankle, pose::LandmarkId::right_ankle}};

}
