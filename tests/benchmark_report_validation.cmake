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
  if(result EQUAL 0)
    message(FATAL_ERROR "Validator accepted corrupted report: ${name}")
  endif()
endfunction()

string(JSON first_sequence GET "${valid_report}" warmup processed_frames 0 sequence)
set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" warmup processed_frames 1 sequence "${first_sequence}")
expect_rejected(frame-order "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" warmup frames_processed 999)
expect_rejected(phase-count "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" measured timings pose_inference samples 0)
expect_rejected(timing-count "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model detector_sha256 "\"bad\"")
expect_rejected(model-hash "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model execution_provider "\"unknown\"")
expect_rejected(provider-identity "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" model execution_policy "\"cuda-audited\"")
expect_rejected(policy-identity "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" measured timings estimator_end_to_end median_ms 999999)
expect_rejected(timing-aggregate "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" status error_count 1)
expect_rejected(success-invariant "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" status success false)
string(JSON corrupted SET "${corrupted}" status error_count 1)
string(JSON corrupted SET "${corrupted}" status terminal_error null)
expect_rejected(terminal-error "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" configuration simulated_source_fps 60)
expect_rejected(simulated-rate-mode "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" input delivery_target_fps 123)
expect_rejected(delivery-rate "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" configuration loop_video true)
expect_rejected(synthetic-loop "${corrupted}")

set(corrupted "${valid_report}")
string(JSON corrupted SET "${corrupted}" configuration recreate_audited_sessions_for_experiment true)
expect_rejected(cpu-session-recreation "${corrupted}")
