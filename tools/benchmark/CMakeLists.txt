add_executable(squatris-pose-benchmark main.cpp parity.cpp report.cpp)
target_link_libraries(squatris-pose-benchmark PRIVATE
  squatris_backends
  squatris_core
  squatris_project_options
  nlohmann_json::nlohmann_json
  opencv_core
)
set_target_properties(squatris-pose-benchmark PROPERTIES
  OUTPUT_NAME "SquatrisPoseBenchmark"
  FOLDER "Squatris/Tools"
)
squatris_stage_onnxruntime(squatris-pose-benchmark)
