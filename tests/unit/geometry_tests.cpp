#include <cmath>

#include <gtest/gtest.h>

#include "squatris/core/pose/geometry.hpp"

namespace {

squatris::pose::Landmark landmark(float x, float y, float confidence = 1.0F) {
  return squatris::pose::make_landmark({x, y}, std::nullopt, confidence);
}

TEST(JointAngle, CalculatesRightAngle) {
  const auto angle = squatris::pose::joint_angle_degrees(landmark(0.5F, 0.0F), landmark(0.5F, 0.5F),
                                                         landmark(1.0F, 0.5F), 0.5F);
  ASSERT_TRUE(angle.has_value());
  EXPECT_NEAR(*angle, 90.0F, 0.001F);
}

TEST(JointAngle, CalculatesStraightJoint) {
  const auto angle = squatris::pose::joint_angle_degrees(landmark(0.5F, 0.0F), landmark(0.5F, 0.5F),
                                                         landmark(0.5F, 1.0F), 0.5F);
  ASSERT_TRUE(angle.has_value());
  EXPECT_NEAR(*angle, 180.0F, 0.001F);
}

TEST(JointAngle, RejectsDegenerateGeometry) {
  const auto angle = squatris::pose::joint_angle_degrees(landmark(0.5F, 0.5F), landmark(0.5F, 0.5F),
                                                         landmark(0.5F, 1.0F), 0.5F);
  EXPECT_FALSE(angle.has_value());
}

TEST(JointAngle, RejectsMissingOrLowConfidenceLandmarks) {
  squatris::pose::Landmark missing;
  EXPECT_FALSE(squatris::pose::joint_angle_degrees(missing, landmark(0.5F, 0.5F),
                                                   landmark(0.5F, 1.0F), 0.5F));
  EXPECT_FALSE(squatris::pose::joint_angle_degrees(landmark(0.5F, 0.0F, 0.2F), landmark(0.5F, 0.5F),
                                                   landmark(0.5F, 1.0F), 0.5F));
}

}
