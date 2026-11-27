find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(squatris-tests
  unit/pose_tests.cpp
  unit/geometry_tests.cpp
  unit/latest_frame_buffer_tests.cpp
  unit/rtmpose_math_tests.cpp
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

add_test(NAME squatris-benchmark-smoke
  COMMAND squatris-pose-benchmark --backend mock --warmup 2 --iterations 5
          --output ${CMAKE_CURRENT_BINARY_DIR}/benchmark-smoke.json
)
set_tests_properties(squatris-benchmark-smoke PROPERTIES LABELS "integration;hardware-independent")

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
      --backend rtmpose
      --provider ${SQUATRIS_HARDWARE_PROVIDER}
      --warmup 2
      --iterations 5
      --detector ${PROJECT_SOURCE_DIR}/models/detector.onnx
      --pose ${PROJECT_SOURCE_DIR}/models/pose.onnx
      --output ${CMAKE_CURRENT_BINARY_DIR}/hardware-provider-smoke.json
  )
  add_test(NAME squatris-hardware-prerecorded-qualification
    COMMAND squatris-pose-benchmark
      --video ${SQUATRIS_HARDWARE_VIDEO}
      --backend rtmpose
      --provider ${SQUATRIS_HARDWARE_PROVIDER}
      --warmup 100
      --iterations 2000
      --detector ${PROJECT_SOURCE_DIR}/models/detector.onnx
      --pose ${PROJECT_SOURCE_DIR}/models/pose.onnx
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
