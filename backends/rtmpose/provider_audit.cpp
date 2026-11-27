#include "squatris/backends/rtmpose/provider_audit.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string_view>

#include <nlohmann/json.hpp>

#include "squatris/core/crypto/sha256.hpp"

namespace squatris::backends::rtmpose {

namespace {

Result<nlohmann::json> load_json(const std::filesystem::path& path) {
  try {
    std::ifstream input(path);
    if (!input) {
      return Status{ErrorCode::not_found, "JSON evidence file was not found: " + path.string()};
    }
    nlohmann::json result;
    input >> result;
    return result;
  } catch (const std::exception& exception) {
    return Status{ErrorCode::invalid_data,
                  "Could not parse JSON evidence " + path.string() + ": " + exception.what()};
  }
}

std::string stable_node_name(std::string name) {
  constexpr std::string_view suffix{"_kernel_time"};
  if (name.ends_with(suffix)) {
    name.resize(name.size() - suffix.size());
  }
  return name;
}

std::string node_key(const nlohmann::json& node) {
  return node.at("name").get<std::string>() + "\n" + node.at("operator").get<std::string>() + "\n" +
         node.at("node_index").get<std::string>();
}

std::uint64_t byte_value(const nlohmann::json& args, std::string_view key) {
  const auto iterator = args.find(key);
  if (iterator == args.end()) {
    return 0;
  }
  if (iterator->is_number_unsigned()) {
    return iterator->get<std::uint64_t>();
  }
  if (iterator->is_number_integer()) {
    const auto value = iterator->get<std::int64_t>();
    return value >= 0 ? static_cast<std::uint64_t>(value) : 0;
  }
  if (iterator->is_string()) {
    try {
      return std::stoull(iterator->get<std::string>());
    } catch (const std::exception&) {
      return 0;
    }
  }
  return 0;
}

bool copy_event(const nlohmann::json& event) {
  std::string text = event.value("name", std::string{});
  if (event.contains("args") && event["args"].is_object()) {
    text += event["args"].value("op_name", std::string{});
  }
  std::ranges::transform(text, text.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return text.find("memcpy") != std::string::npos || text.find("copy") != std::string::npos ||
         text.find("transfer") != std::string::npos;
}

Result<nlohmann::json> analyze_profile(const ProviderProfile& profile) {
  const auto trace = load_json(profile.path);
  if (!trace) {
    return trace.error();
  }
  if (!trace.value().is_array()) {
    return Status{ErrorCode::invalid_data,
                  "ONNX Runtime profile is not an event array: " + profile.path.string()};
  }
  std::map<std::string, nlohmann::json> nodes;
  nlohmann::json memory_copies = nlohmann::json::array();
  for (const auto& event : trace.value()) {
    if (!event.is_object()) {
      continue;
    }
    if (copy_event(event)) {
      memory_copies.push_back(event);
    }
    const auto event_name = event.value("name", std::string{});
    const auto is_kernel_record = event_name.ends_with("_kernel_time");
    if (!event.contains("args") || !event["args"].is_object()) {
      if (is_kernel_record) {
        return Status{ErrorCode::invalid_data,
                      "Provider profile contains a kernel record without node arguments"};
      }
      continue;
    }
    const auto& args = event["args"];
    const auto has_provider_field = args.contains("provider");
    if (!is_kernel_record && !has_provider_field) {
      continue;
    }
    if (!is_kernel_record || !args.contains("provider") || !args["provider"].is_string() ||
        args["provider"].get_ref<const std::string&>().empty() || !args.contains("op_name") ||
        !args["op_name"].is_string() || args["op_name"].get_ref<const std::string&>().empty() ||
        !args.contains("node_index") || !args["node_index"].is_string() ||
        args["node_index"].get_ref<const std::string&>().empty() ||
        !args.contains("input_type_shape") || !args.contains("output_type_shape") ||
        !event.contains("dur") || !event["dur"].is_number_unsigned()) {
      return Status{ErrorCode::invalid_data,
                    "Provider profile contains a malformed or incomplete node record"};
    }
    const auto name = stable_node_name(event_name);
    if (name.empty()) {
      return Status{ErrorCode::invalid_data, "Provider profile contains an unnamed node record"};
    }
    const nlohmann::json identity = {
        {"name", name}, {"operator", args["op_name"]}, {"node_index", args["node_index"]}};
    const auto key = node_key(identity);
    auto iterator = nodes.find(key);
    const auto duration = event.value("dur", std::uint64_t{});
    if (iterator == nodes.end()) {
      nlohmann::json record = identity;
      record["provider"] = args["provider"];
      record["input_type_shape"] = args.value("input_type_shape", nlohmann::json::array());
      record["output_type_shape"] = args.value("output_type_shape", nlohmann::json::array());
      record["parameter_bytes"] = byte_value(args, "parameter_size");
      record["activation_bytes"] = byte_value(args, "activation_size");
      record["output_bytes"] = byte_value(args, "output_size");
      record["execution_count"] = 1;
      record["total_duration_us"] = duration;
      record["minimum_duration_us"] = duration;
      record["maximum_duration_us"] = duration;
      nodes.emplace(key, std::move(record));
    } else {
      auto& record = iterator->second;
      if (record["provider"] != args["provider"] || record["operator"] != args["op_name"]) {
        return Status{ErrorCode::invalid_data,
                      "Provider profile contains an inconsistent stable node identity"};
      }
      record["execution_count"] = record["execution_count"].get<std::uint64_t>() + 1;
      record["total_duration_us"] = record["total_duration_us"].get<std::uint64_t>() + duration;
      record["minimum_duration_us"] =
          std::min(record["minimum_duration_us"].get<std::uint64_t>(), duration);
      record["maximum_duration_us"] =
          std::max(record["maximum_duration_us"].get<std::uint64_t>(), duration);
    }
  }
  nlohmann::json cuda_nodes = nlohmann::json::array();
  nlohmann::json cpu_nodes = nlohmann::json::array();
  nlohmann::json unknown_nodes = nlohmann::json::array();
  for (const auto& [key, node] : nodes) {
    static_cast<void>(key);
    const auto provider = node.at("provider").get<std::string>();
    if (provider == "CUDAExecutionProvider") {
      cuda_nodes.push_back(node);
    } else if (provider == "CPUExecutionProvider") {
      cpu_nodes.push_back(node);
    } else {
      unknown_nodes.push_back(node);
    }
  }
  const auto hash = crypto::sha256_file(profile.path);
  if (!hash) {
    return hash.error();
  }
  return nlohmann::json{{"session", profile.session},
                        {"path", std::filesystem::absolute(profile.path).string()},
                        {"sha256", hash.value()},
                        {"cuda_nodes", cuda_nodes},
                        {"cpu_nodes", cpu_nodes},
                        {"unknown_provider_nodes", unknown_nodes},
                        {"memory_copy_events", memory_copies}};
}

std::set<std::string> identities(const nlohmann::json& nodes) {
  std::set<std::string> result;
  for (const auto& node : nodes) {
    result.insert(node_key(node));
  }
  return result;
}

nlohmann::json selected_nodes(const nlohmann::json& nodes, const std::set<std::string>& keys) {
  nlohmann::json result = nlohmann::json::array();
  for (const auto& node : nodes) {
    if (keys.contains(node_key(node))) {
      result.push_back(node);
    }
  }
  return result;
}

}

Result<nlohmann::json> analyze_provider_profiles(const std::vector<ProviderProfile>& profiles,
                                                 const std::filesystem::path& policy_manifest,
                                                 const std::string& detector_sha256,
                                                 const std::string& pose_sha256,
                                                 const std::string& runtime_version,
                                                 const std::string& runtime_header_version,
                                                 const nlohmann::json& runtime_artifacts,
                                                 bool verified_sessions_retained) {
  const auto manifest = load_json(policy_manifest);
  if (!manifest) {
    return manifest.error();
  }
  const auto manifest_hash = crypto::sha256_file(policy_manifest);
  if (!manifest_hash) {
    return manifest_hash.error();
  }
  try {
    nlohmann::json errors = nlohmann::json::array();
    if (manifest.value().at("schema_version") != 1 ||
        manifest.value().at("runtime_version") != runtime_version ||
        manifest.value().at("graph_optimization_level") != "ORT_ENABLE_ALL" ||
        manifest.value().at("execution_mode") != "ORT_SEQUENTIAL") {
      errors.push_back("Policy identity does not match the configured ONNX Runtime session");
    }
    if (runtime_header_version != runtime_version) {
      errors.push_back("Loaded ONNX Runtime version does not match the compile-time headers");
    }
    if (!runtime_artifacts.is_array() || runtime_artifacts.empty()) {
      errors.push_back("Loaded ONNX Runtime artifacts were not identified");
    }
    std::map<std::string, nlohmann::json> analyzed;
    for (const auto& profile : profiles) {
      if (profile.session != "detector" && profile.session != "pose") {
        errors.push_back("Profile identifies an unknown session: " + profile.session);
        continue;
      }
      if (analyzed.contains(profile.session)) {
        errors.push_back("Provider evidence contains a duplicate session: " + profile.session);
        continue;
      }
      const auto result = analyze_profile(profile);
      if (!result) {
        return result.error();
      }
      analyzed.emplace(profile.session, result.value());
    }
    nlohmann::json sessions = nlohmann::json::array();
    for (const auto* session_name : {"detector", "pose"}) {
      const auto analyzed_iterator = analyzed.find(session_name);
      if (analyzed_iterator == analyzed.end()) {
        errors.push_back(std::string("Provider evidence is missing session: ") + session_name);
        continue;
      }
      auto session = analyzed_iterator->second;
      const auto& expected = manifest.value().at("sessions").at(session_name);
      const auto actual_model_hash =
          std::string_view(session_name) == "detector" ? detector_sha256 : pose_sha256;
      if (expected.at("model_sha256") != actual_model_hash) {
        errors.push_back(std::string("Model hash does not match policy for session: ") +
                         session_name);
      }
      const auto approved = identities(expected.at("approved_cpu_nodes"));
      const auto actual_cpu = identities(session.at("cpu_nodes"));
      std::set<std::string> unexpected;
      std::set<std::string> missing;
      std::ranges::set_difference(actual_cpu, approved,
                                  std::inserter(unexpected, unexpected.begin()));
      std::ranges::set_difference(approved, actual_cpu, std::inserter(missing, missing.begin()));
      session["approved_cpu_nodes"] = selected_nodes(session.at("cpu_nodes"), approved);
      session["unexpected_cpu_nodes"] = selected_nodes(session.at("cpu_nodes"), unexpected);
      nlohmann::json missing_nodes = nlohmann::json::array();
      for (const auto& node : expected.at("approved_cpu_nodes")) {
        if (missing.contains(node_key(node))) {
          missing_nodes.push_back(node);
        }
      }
      session["missing_approved_cpu_nodes"] = missing_nodes;
      const auto expected_cuda_count = expected.at("expected_cuda_node_count").get<std::size_t>();
      const auto expected_total_count =
          expected.at("expected_optimized_node_count").get<std::size_t>();
      const auto actual_total = session.at("cuda_nodes").size() + session.at("cpu_nodes").size() +
                                session.at("unknown_provider_nodes").size();
      if (session.at("cuda_nodes").size() != expected_cuda_count ||
          actual_total != expected_total_count) {
        errors.push_back(std::string("Optimized node counts do not match policy for session: ") +
                         session_name);
      }
      if (!unexpected.empty() || !missing.empty()) {
        errors.push_back(std::string("CPU node assignments do not match policy for session: ") +
                         session_name);
      }
      if (!session.at("unknown_provider_nodes").empty()) {
        errors.push_back(std::string("Unknown execution provider observed in session: ") +
                         session_name);
      }
      const auto cuda_identities = identities(session.at("cuda_nodes"));
      for (const auto& required : expected.at("required_cuda_nodes")) {
        if (!cuda_identities.contains(node_key(required))) {
          errors.push_back(
              std::string("Required CUDA neural-compute node is missing in session: ") +
              session_name + ": " + required.at("name").get<std::string>());
        }
      }
      session["model_sha256"] = actual_model_hash;
      session["static_graph"] = expected.at("static_graph");
      session["verified"] = unexpected.empty() && missing.empty() &&
                            session.at("unknown_provider_nodes").empty() &&
                            session.at("cuda_nodes").size() == expected_cuda_count &&
                            actual_total == expected_total_count;
      sessions.push_back(std::move(session));
    }
    return nlohmann::json{{"method", "onnxruntime-profile-plus-pinned-policy"},
                          {"policy_id", manifest.value().at("policy_id")},
                          {"policy_manifest", std::filesystem::absolute(policy_manifest).string()},
                          {"policy_manifest_sha256", manifest_hash.value()},
                          {"runtime_version", runtime_version},
                          {"runtime_header_version", runtime_header_version},
                          {"runtime_artifacts", runtime_artifacts},
                          {"verified_sessions_retained", verified_sessions_retained},
                          {"graph_optimization_level", "ORT_ENABLE_ALL"},
                          {"execution_mode", "ORT_SEQUENTIAL"},
                          {"cuda_graph_execution", false},
                          {"static_assignment_complete", false},
                          {"runtime_evidence_scope", "observed-straight-line-preflight"},
                          {"sessions", sessions},
                          {"verification_errors", errors},
                          {"verified", errors.empty()}};
  } catch (const nlohmann::json::exception& exception) {
    return Status{ErrorCode::invalid_data,
                  "Provider policy manifest is invalid: " + std::string(exception.what())};
  }
}

}
