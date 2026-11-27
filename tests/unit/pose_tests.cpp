#include <array>
#include <chrono>
#include <limits>

#include <gtest/gtest.h>

#include "squatris/core/pose/mapping.hpp"
#include "squatris/core/pose/pose.hpp"

namespace {

using squatris::ErrorCode;
using squatris::pose::LandmarkAvailability;
using squatris::pose::LandmarkId;
using squatris::pose::ModelLandmark;

TEST(PoseCoordinates, NormalizesPixelCoordinates) {
  const auto point = squatris::pose::normalize_image_point(960.0F, 540.0F, {1920, 1080});
  ASSERT_TRUE(point);
  EXPECT_FLOAT_EQ(point.value().x, 0.5F);
  EXPECT_FLOAT_EQ(point.value().y, 0.5F);
}

TEST(PoseCoordinates, RejectsInvalidDimensionsAndNonFiniteCoordinates) {
  const auto zero_size = squatris::pose::normalize_image_point(1.0F, 1.0F, {0, 1080});
  ASSERT_FALSE(zero_size);
  EXPECT_EQ(zero_size.error().code, ErrorCode::invalid_argument);
  const auto non_finite = squatris::pose::normalize_image_point(
      std::numeric_limits<float>::quiet_NaN(), 1.0F, {1920, 1080});
  ASSERT_FALSE(non_finite);
  EXPECT_EQ(non_finite.error().code, ErrorCode::invalid_data);
}

TEST(PoseCoordinates, RejectsPointsOutsideTheImage) {
  EXPECT_FALSE(squatris::pose::normalize_image_point(-1.0F, 4.0F, {10, 10}));
  EXPECT_FALSE(squatris::pose::normalize_image_point(4.0F, 11.0F, {10, 10}));
}

TEST(PoseLandmark, InvalidConfidenceProducesUnavailableLandmark) {
  const auto invalid = squatris::pose::make_landmark({0.5F, 0.5F}, std::nullopt, 1.1F);
  EXPECT_EQ(invalid.availability, LandmarkAvailability::unavailable);
  EXPECT_FALSE(invalid.image.has_value());
  EXPECT_FALSE(invalid.confidence.has_value());
  const auto non_finite = squatris::pose::make_landmark({0.5F, 0.5F}, std::nullopt,
                                                        std::numeric_limits<float>::infinity());
  EXPECT_EQ(non_finite.availability, LandmarkAvailability::unavailable);
}

TEST(PoseMapping, PreservesAnatomicalLeftAndRight) {
  std::array<ModelLandmark, 17> source{};
  for (auto& landmark : source) {
    landmark = {500.0F, 500.0F, 0.9F};
  }
  source[5] = {100.0F, 300.0F, 0.95F};
  source[6] = {900.0F, 300.0F, 0.96F};
  const auto mapped =
      squatris::pose::map_landmarks(squatris::pose::LandmarkLayout::coco17, source, {1000, 1000});
  ASSERT_TRUE(mapped);
  const auto& left = mapped.value()[static_cast<std::size_t>(LandmarkId::left_shoulder)];
  const auto& right = mapped.value()[static_cast<std::size_t>(LandmarkId::right_shoulder)];
  ASSERT_TRUE(left.image.has_value());
  ASSERT_TRUE(right.image.has_value());
  EXPECT_FLOAT_EQ(left.image.value_or(squatris::pose::NormalizedImagePoint{}).x, 0.1F);
  EXPECT_FLOAT_EQ(right.image.value_or(squatris::pose::NormalizedImagePoint{}).x, 0.9F);
}

TEST(PoseMapping, LeavesUnsupportedFeetExplicitlyUnavailable) {
  std::array<ModelLandmark, 17> source{};
  for (auto& landmark : source) {
    landmark = {100.0F, 100.0F, 0.9F};
  }
  const auto mapped =
      squatris::pose::map_landmarks(squatris::pose::LandmarkLayout::coco17, source, {200, 200});
  ASSERT_TRUE(mapped);
  EXPECT_EQ(mapped.value()[static_cast<std::size_t>(LandmarkId::left_heel)].availability,
            LandmarkAvailability::unavailable);
  EXPECT_EQ(mapped.value()[static_cast<std::size_t>(LandmarkId::right_foot_index)].availability,
            LandmarkAvailability::unavailable);
}

TEST(PoseMapping, RejectsLayoutWithTooFewLandmarks) {
  std::array<ModelLandmark, 5> source{};
  const auto mapped =
      squatris::pose::map_landmarks(squatris::pose::LandmarkLayout::halpe26, source, {200, 200});
  ASSERT_FALSE(mapped);
  EXPECT_EQ(mapped.error().code, ErrorCode::invalid_data);
}

TEST(PoseMapping, MapsPinnedHalpeFeetAndHeels) {
  std::array<ModelLandmark, 26> source{};
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = {static_cast<float>(index * 10), 100.0F, 0.9F};
  }
  const auto mapped =
      squatris::pose::map_landmarks(squatris::pose::LandmarkLayout::halpe26, source, {300, 200});
  ASSERT_TRUE(mapped);
  const auto x_for = [&mapped](LandmarkId id) {
    return mapped.value()[static_cast<std::size_t>(id)]
        .image.value_or(squatris::pose::NormalizedImagePoint{})
        .x;
  };
  EXPECT_FLOAT_EQ(x_for(LandmarkId::left_foot_index), 200.0F / 300.0F);
  EXPECT_FLOAT_EQ(x_for(LandmarkId::right_foot_index), 210.0F / 300.0F);
  EXPECT_FLOAT_EQ(x_for(LandmarkId::left_heel), 240.0F / 300.0F);
  EXPECT_FLOAT_EQ(x_for(LandmarkId::right_heel), 250.0F / 300.0F);
}

TEST(PoseValidation, RejectsTimestampInversion) {
  squatris::pose::PoseFrame pose;
  pose.image_size = {640, 480};
  pose.captured_at = squatris::pose::Clock::now();
  pose.inference_completed_at = pose.captured_at - std::chrono::milliseconds(1);
  pose.inference_backend = "test";
  const auto status = squatris::pose::validate_pose(pose);
  EXPECT_EQ(status.code, ErrorCode::invalid_data);
}

TEST(PoseValidation, AcceptsExplicitMissingLandmarks) {
  squatris::pose::PoseFrame pose;
  pose.image_size = {640, 480};
  pose.captured_at = squatris::pose::Clock::now();
  pose.inference_completed_at = pose.captured_at;
  pose.inference_backend = "test";
  EXPECT_TRUE(squatris::pose::validate_pose(pose).ok());
}

}
