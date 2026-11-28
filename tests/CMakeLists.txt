find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(squatris-tests
  unit/pose_tests.cpp
  unit/geometry_tests.cpp
  unit/latest_frame_buffer_tests.cpp
  unit/tracking_assessment_tests.cpp
  unit/rtmpose_math_tests.cpp
  unit/provider_audit_tests.cpp
  unit/atomic_file_tests.cpp
  integration/pipeline_tests.cpp
)
target_link_libraries(squatris-tests PRIVATE
  squatris_backends
  squatris_core
  squatris_project_options
  GTest::gtest_main
)
set_target_properties(squatris-tests PROPERTIES FOLDER "Squatris/Tests")
squatris_stage_onnxruntime(squatris-tests)
gtest_discover_tests(squatris-tests)

add_test(NAME squatris-benchmark-deterministic-smoke
  COMMAND squatris-pose-benchmark --mode deterministic --backend mock --warmup 2 --iterations 5
          --output ${CMAKE_CURRENT_BINARY_DIR}/benchmark-deterministic-smoke.json
)
add_test(NAME squatris-benchmark-deterministic-report-validation
  COMMAND squatris-pose-benchmark --validate-report
          ${CMAKE_CURRENT_BINARY_DIR}/benchmark-deterministic-smoke.json
)
add_test(NAME squatris-benchmark-report-corruption-rejection
  COMMAND ${CMAKE_COMMAND}
          -DBENCHMARK=$<TARGET_FILE:squatris-pose-benchmark>
          -DREPORT=${CMAKE_CURRENT_BINARY_DIR}/benchmark-deterministic-smoke.json
          -P ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_report_validation.cmake
)
add_test(NAME squatris-benchmark-realtime-smoke
  COMMAND squatris-pose-benchmark --mode realtime --backend mock --warmup 2 --iterations 5
          --output ${CMAKE_CURRENT_BINARY_DIR}/benchmark-realtime-smoke.json
)
add_test(NAME squatris-benchmark-realtime-report-validation
  COMMAND squatris-pose-benchmark --validate-report
          ${CMAKE_CURRENT_BINARY_DIR}/benchmark-realtime-smoke.json
)
add_test(NAME squatris-benchmark-rejects-nonfinite-source-rate
  COMMAND squatris-pose-benchmark --mode realtime --backend mock --video unused.mp4
          --simulated-source-fps nan
)
set_tests_properties(squatris-benchmark-rejects-nonfinite-source-rate PROPERTIES
  LABELS "integration;hardware-independent"
  WILL_FAIL TRUE
)
add_test(NAME squatris-benchmark-rejects-unconfigured-rtmw
  COMMAND squatris-pose-benchmark --backend rtmw --provider cpu
)
set_tests_properties(squatris-benchmark-rejects-unconfigured-rtmw PROPERTIES
  LABELS "integration;hardware-independent"
  WILL_FAIL TRUE
)
add_test(NAME squatris-benchmark-rejects-unknown-provider
  COMMAND squatris-pose-benchmark --mode deterministic --backend mock --provider unknown
)
add_test(NAME squatris-benchmark-failure-report
  COMMAND squatris-pose-benchmark --backend rtmpose
          --detector ${CMAKE_CURRENT_BINARY_DIR}/missing-detector.onnx
          --pose ${CMAKE_CURRENT_BINARY_DIR}/missing-pose.onnx
          --output ${CMAKE_CURRENT_BINARY_DIR}/benchmark-failure.json
)
add_test(NAME squatris-benchmark-failure-report-validation
  COMMAND squatris-pose-benchmark --validate-report
          ${CMAKE_CURRENT_BINARY_DIR}/benchmark-failure.json
)
set_tests_properties(squatris-benchmark-deterministic-smoke PROPERTIES
  LABELS "integration;hardware-independent"
  FIXTURES_SETUP deterministic_benchmark_report
)
set_tests_properties(squatris-benchmark-deterministic-report-validation PROPERTIES
  LABELS "integration;hardware-independent"
  FIXTURES_REQUIRED deterministic_benchmark_report
)
set_tests_properties(squatris-benchmark-report-corruption-rejection PROPERTIES
  LABELS "integration;hardware-independent"
  FIXTURES_REQUIRED deterministic_benchmark_report
)
set_tests_properties(squatris-benchmark-realtime-smoke PROPERTIES
  LABELS "integration;hardware-independent"
  FIXTURES_SETUP realtime_benchmark_report
)
set_tests_properties(squatris-benchmark-realtime-report-validation PROPERTIES
  LABELS "integration;hardware-independent"
  FIXTURES_REQUIRED realtime_benchmark_report
)
set_tests_properties(squatris-benchmark-rejects-unknown-provider PROPERTIES
  LABELS "integration;hardware-independent"
  WILL_FAIL TRUE
)
set_tests_properties(squatris-benchmark-failure-report PROPERTIES
  LABELS "integration;hardware-independent"
  WILL_FAIL TRUE
  FIXTURES_SETUP failed_benchmark_report
)
set_tests_properties(squatris-benchmark-failure-report-validation PROPERTIES
  LABELS "integration;hardware-independent"
  FIXTURES_REQUIRED failed_benchmark_report
)

if(SQUATRIS_ENABLE_GUI_SMOKE_TEST)
  if(NOT WIN32)
    message(FATAL_ERROR "The native Pose Lab GUI smoke test currently requires Windows")
  endif()
  add_test(NAME squatris-pose-lab-gui-smoke
    COMMAND squatris-pose-lab --synthetic --backend mock --width 640 --height 480 --fps 30
            --gui-smoke-seconds 3
  )
  set_tests_properties(squatris-pose-lab-gui-smoke PROPERTIES
    LABELS "gui-smoke;opt-in"
    TIMEOUT 15
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endif()

if(SQUATRIS_ENABLE_HARDWARE_TESTS)
  if(NOT SQUATRIS_ENABLE_ONNXRUNTIME)
    message(FATAL_ERROR "Hardware tests require SQUATRIS_ENABLE_ONNXRUNTIME=ON")
  endif()
  if(NOT SQUATRIS_BUILD_BENCHMARKS)
    message(FATAL_ERROR "Hardware tests require SQUATRIS_BUILD_BENCHMARKS=ON")
  endif()
  if(NOT EXISTS "${SQUATRIS_HARDWARE_VIDEO}")
    message(FATAL_ERROR "Set SQUATRIS_HARDWARE_VIDEO to a qualification recording")
  endif()
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/models/detector.onnx" OR
     NOT EXISTS "${PROJECT_SOURCE_DIR}/models/pose.onnx")
    message(FATAL_ERROR "Run scripts/acquire-models.ps1 before enabling hardware tests")
  endif()

  add_test(NAME squatris-hardware-provider-smoke
    COMMAND squatris-pose-benchmark
      --video ${SQUATRIS_HARDWARE_VIDEO}
      --backend rtmpose
      --provider ${SQUATRIS_HARDWARE_PROVIDER}
      --mode deterministic
      --warmup 2
      --iterations 5
      --detector ${PROJECT_SOURCE_DIR}/models/detector.onnx
      --pose ${PROJECT_SOURCE_DIR}/models/pose.onnx
      --manifest ${PROJECT_SOURCE_DIR}/models/manifest.json
      --policy-manifest ${PROJECT_SOURCE_DIR}/config/rtmpose-cuda-policy.json
      --output ${CMAKE_CURRENT_BINARY_DIR}/hardware-provider-smoke.json
  )
  add_test(NAME squatris-hardware-prerecorded-qualification
    COMMAND squatris-pose-benchmark
      --video ${SQUATRIS_HARDWARE_VIDEO}
      --backend rtmpose
      --provider ${SQUATRIS_HARDWARE_PROVIDER}
      --mode deterministic
      --warmup 100
      --iterations 2000
      --detector ${PROJECT_SOURCE_DIR}/models/detector.onnx
      --pose ${PROJECT_SOURCE_DIR}/models/pose.onnx
      --manifest ${PROJECT_SOURCE_DIR}/models/manifest.json
      --policy-manifest ${PROJECT_SOURCE_DIR}/config/rtmpose-cuda-policy.json
      --output ${CMAKE_CURRENT_BINARY_DIR}/hardware-prerecorded-qualification.json
  )
  set_tests_properties(
    squatris-hardware-provider-smoke
    squatris-hardware-prerecorded-qualification
    PROPERTIES
      LABELS "hardware-required;qualification"
      TIMEOUT 1800
  )
endif()
