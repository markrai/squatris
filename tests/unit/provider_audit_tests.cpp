#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "squatris/backends/rtmpose/provider_audit.hpp"

namespace {

using nlohmann::json;
using squatris::backends::rtmpose::ProviderProfile;
using squatris::backends::rtmpose::analyze_provider_profiles;

#ifdef _WIN32

class ProviderAuditTest : public testing::Test {
protected:
  void SetUp() override {
    directory_ = std::filesystem::temp_directory_path() /
                 ("squatris-audit-" + std::to_string(
                                         std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
    const auto cpu_identity = json{{"name", "ShapeGather"},
                                   {"operator", "Gather"},
                                   {"node_index", "1"},
                                   {"input_type_shape", shape()},
                                   {"output_type_shape", shape()}};
    const auto cuda_identity = json{{"name", "Neural"},
                                    {"operator", "Conv"},
                                    {"node_index", "0"}};
    const auto session = json{{"model_sha256", model_hash_},
                              {"expected_optimized_node_count", 2},
                              {"expected_cuda_node_count", 1},
                              {"static_graph", json::object()},
                              {"required_cuda_nodes", json::array({cuda_identity})},
                              {"approved_cpu_nodes", json::array({cpu_identity})}};
    write("policy.json", {{"schema_version", 1},
                          {"policy_id", "fixture"},
                          {"runtime_version", "1.20.1"},
                          {"graph_optimization_level", "ORT_ENABLE_ALL"},
                          {"execution_mode", "ORT_SEQUENTIAL"},
                          {"sessions", {{"detector", session}, {"pose", session}}}});
    valid_trace_ = json::array({event("Neural", "Conv", "0", "CUDAExecutionProvider"),
                               event("ShapeGather", "Gather", "1", "CPUExecutionProvider")});
    write("detector.json", valid_trace_);
    write("pose.json", valid_trace_);
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
    }
  }

  static json shape() { return json::array({json{{"int64", json::array({1})}}}); }

  static json event(const std::string& name, const std::string& op, const std::string& index,
                    const std::string& provider) {
    return {{"name", name + "_kernel_time"},
            {"dur", 1U},
            {"args", {{"provider", provider},
                      {"op_name", op},
                      {"node_index", index},
                      {"input_type_shape", shape()},
                      {"output_type_shape", shape()}}}};
  }

  void write(const std::string& name, const json& data) const {
    std::ofstream stream(directory_ / name, std::ios::binary);
    stream << data.dump();
    stream.close();
    ASSERT_TRUE(stream.good());
  }

  auto audit() const {
    const std::vector<ProviderProfile> profiles{{"detector", directory_ / "detector.json"},
                                                {"pose", directory_ / "pose.json"}};
    return analyze_provider_profiles(profiles, directory_ / "policy.json", model_hash_, model_hash_,
                                     "1.20.1", "1.20.1", json::array({"fixture"}), true);
  }

  std::filesystem::path directory_;
  json valid_trace_;
  const std::string model_hash_ = std::string(64, 'a');
};

TEST_F(ProviderAuditTest, AcceptsExactObservedAssignment) {
  const auto result = audit();
  ASSERT_TRUE(result);
  EXPECT_TRUE(result.value().at("verified").get<bool>());
  EXPECT_TRUE(result.value().at("sessions").at(0).at("verified").get<bool>());
  EXPECT_FALSE(result.value().at("static_assignment_complete").get<bool>());
}

TEST_F(ProviderAuditTest, RejectsPinnedCpuShapeMismatch) {
  valid_trace_[1]["args"]["input_type_shape"] = json::array({json{{"float", json::array({64})}}});
  write("detector.json", valid_trace_);
  const auto result = audit();
  ASSERT_TRUE(result);
  EXPECT_FALSE(result.value().at("verified").get<bool>());
  EXPECT_FALSE(result.value().at("sessions").at(0).at("verified").get<bool>());
  EXPECT_TRUE(result.value().at("sessions").at(1).at("verified").get<bool>());
}

TEST_F(ProviderAuditTest, RejectsCpuShapeDriftAcrossEvents) {
  auto changed = valid_trace_.at(1);
  changed["args"]["output_type_shape"] = json::array({json{{"int64", json::array({2})}}});
  valid_trace_.push_back(changed);
  write("detector.json", valid_trace_);
  EXPECT_FALSE(audit());
}

TEST_F(ProviderAuditTest, RejectsMalformedShapeEvidence) {
  valid_trace_[1]["args"]["input_type_shape"] = nullptr;
  write("detector.json", valid_trace_);
  EXPECT_FALSE(audit());
}

TEST_F(ProviderAuditTest, RejectsNonObjectProfileEvents) {
  valid_trace_.push_back(nullptr);
  write("detector.json", valid_trace_);
  EXPECT_FALSE(audit());
}

TEST_F(ProviderAuditTest, MissingRequiredCudaNodeInvalidatesSession) {
  valid_trace_[0]["name"] = "OtherNeural_kernel_time";
  write("detector.json", valid_trace_);
  const auto result = audit();
  ASSERT_TRUE(result);
  EXPECT_FALSE(result.value().at("verified").get<bool>());
  EXPECT_FALSE(result.value().at("sessions").at(0).at("verified").get<bool>());
}

#endif

}
