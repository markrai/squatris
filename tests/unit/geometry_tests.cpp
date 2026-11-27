#include <cmath>

#include <gtest/gtest.h>

#include "squatris/core/pose/geometry.hpp"

namespace {

squatris::pose::Landmark landmark(float x, float y, float confidence = 1.0F) {
  return squatris::pose::make_landmark({x, y}, std::nullopt, confidence);
}

TEST(JointAngle, CalculatesRightAngle) {
  const auto angle = squatris::pose::joint_angle_degrees(landmark(0.5F, 0.0F), landmark(0.5F, 0.5F),
                                                         landmark(1.0F, 0.5F), {100, 100}, 0.5F);
  ASSERT_TRUE(angle.has_value());
  EXPECT_NEAR(*angle, 90.0F, 0.001F);
}

TEST(JointAngle, CalculatesStraightJoint) {
  const auto angle = squatris::pose::joint_angle_degrees(landmark(0.5F, 0.0F), landmark(0.5F, 0.5F),
                                                         landmark(0.5F, 1.0F), {100, 100}, 0.5F);
  ASSERT_TRUE(angle.has_value());
  EXPECT_NEAR(*angle, 180.0F, 0.001F);
}

TEST(JointAngle, RejectsDegenerateGeometry) {
  const auto angle = squatris::pose::joint_angle_degrees(landmark(0.5F, 0.5F), landmark(0.5F, 0.5F),
                                                         landmark(0.5F, 1.0F), {100, 100}, 0.5F);
  EXPECT_FALSE(angle.has_value());
}

TEST(JointAngle, RejectsMissingOrLowConfidenceLandmarks) {
  squatris::pose::Landmark missing;
  EXPECT_FALSE(squatris::pose::joint_angle_degrees(missing, landmark(0.5F, 0.5F),
                                                   landmark(0.5F, 1.0F), {100, 100}, 0.5F));
  EXPECT_FALSE(squatris::pose::joint_angle_degrees(landmark(0.5F, 0.0F, 0.2F), landmark(0.5F, 0.5F),
                                                   landmark(0.5F, 1.0F), {100, 100}, 0.5F));
}

TEST(JointAngle, PreservesEquivalentPixelGeometryAcrossAspectRatios) {
  const auto angle_for = [](squatris::pose::ImageSize size) {
    const auto first =
        landmark(20.0F / static_cast<float>(size.width), 10.0F / static_cast<float>(size.height));
    const auto vertex =
        landmark(40.0F / static_cast<float>(size.width), 30.0F / static_cast<float>(size.height));
    const auto third =
        landmark(70.0F / static_cast<float>(size.width), 20.0F / static_cast<float>(size.height));
    return squatris::pose::joint_angle_degrees(first, vertex, third, size, 0.5F);
  };
  const auto square = angle_for({100, 100});
  const auto widescreen = angle_for({160, 90});
  const auto portrait = angle_for({90, 160});
  ASSERT_TRUE(square.has_value());
  ASSERT_TRUE(widescreen.has_value());
  ASSERT_TRUE(portrait.has_value());
  EXPECT_NEAR(*square, *widescreen, 0.001F);
  EXPECT_NEAR(*square, *portrait, 0.001F);
}

TEST(JointAngle, RejectsMissingImageDimensions) {
  EXPECT_FALSE(squatris::pose::joint_angle_degrees(landmark(0.0F, 0.0F), landmark(0.5F, 0.5F),
                                                   landmark(1.0F, 0.0F), {}, 0.5F));
}

}
