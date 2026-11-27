#include <array>

#include <gtest/gtest.h>

#include "squatris/backends/rtmpose/model_math.hpp"

namespace {

TEST(RtmposePreprocessing, MatchesUpstreamCenterScaleAndAspectRule) {
  const auto region =
      squatris::backends::rtmpose::prepare_pose_region({0.0F, 0.0F, 100.0F, 200.0F});
  ASSERT_TRUE(region);
  EXPECT_FLOAT_EQ(region.value().center_x, 50.0F);
  EXPECT_FLOAT_EQ(region.value().center_y, 100.0F);
  EXPECT_FLOAT_EQ(region.value().width, 187.5F);
  EXPECT_FLOAT_EQ(region.value().height, 250.0F);
}

TEST(RtmposePreprocessing, RejectsDegeneratePersonBox) {
  EXPECT_FALSE(squatris::backends::rtmpose::prepare_pose_region({1.0F, 2.0F, 1.0F, 5.0F}));
}

TEST(RtmposeDecoding, MatchesUpstreamArgmaxMeanScoreFixture) {
  constexpr std::array x{0.1F, 0.2F, 0.9F, 0.3F};
  constexpr std::array y{0.2F, 0.1F, 0.3F, 0.4F, 0.1F, 0.8F, 0.2F, 0.1F};
  const auto decoded = squatris::backends::rtmpose::decode_simcc(x, y, 1, x.size(), y.size(),
                                                                 {96.0F, 128.0F, 192.0F, 256.0F});
  ASSERT_TRUE(decoded);
  ASSERT_EQ(decoded.value().size(), 1);
  EXPECT_FLOAT_EQ(decoded.value()[0].pixel_x, 1.0F);
  EXPECT_FLOAT_EQ(decoded.value()[0].pixel_y, 2.5F);
  EXPECT_FLOAT_EQ(decoded.value()[0].confidence, 0.85F);
}

TEST(RtmposeDecoding, RejectsMismatchedTensorShape) {
  constexpr std::array x{0.1F, 0.2F};
  constexpr std::array y{0.1F, 0.2F};
  EXPECT_FALSE(
      squatris::backends::rtmpose::decode_simcc(x, y, 2, 2, 2, {96.0F, 128.0F, 192.0F, 256.0F}));
}

}
