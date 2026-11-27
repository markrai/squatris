if(NOT DEFINED BENCHMARK OR NOT DEFINED REPORT)
  message(FATAL_ERROR "BENCHMARK and REPORT are required")
endif()

file(READ "${REPORT}" valid_report)

function(expect_rejected name content)
  set(path "${REPORT}.${name}.json")
  file(WRITE "${path}" "${content}")
  execute_process(
    COMMAND "${BENCHMARK}" --validate-report "${path}"
    RESULT_VARIABLE result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  file(REMOVE "${path}")
  if(result EQUAL 0)
    message(FATAL_ERROR "Validator accepted corrupted provider evidence: ${name}")
  endif()
endfunction()

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model provider_evidence policy_id "\"corrupted\"")
expect_rejected(policy-identity "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model provider_evidence policy_manifest_sha256 "\"bad\"")
expect_rejected(policy-hash "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model provider_evidence sessions 0 sha256 "\"bad\"")
expect_rejected(profile-hash "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model provider_evidence sessions 0 cpu_nodes 0 name "\"corrupted\"")
expect_rejected(node-assignment "${corrupted}")

set(corrupted "${valid_report}")
string(JSON retained GET "${valid_report}" model provider_evidence verified_sessions_retained)
if(retained)
  string(JSON corrupted SET "${corrupted}" model provider_evidence verified_sessions_retained false)
else()
  string(JSON corrupted SET "${corrupted}" model provider_evidence verified_sessions_retained true)
endif()
expect_rejected(session-retention "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model provider_evidence runtime_artifacts 0 name "\"not-a-runtime.dll\"")
expect_rejected(runtime-module-identity "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model provider_evidence static_assignment_complete true)
expect_rejected(static-assignment-overclaim "${corrupted}")
