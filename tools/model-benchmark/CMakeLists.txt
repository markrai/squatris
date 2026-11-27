add_executable(squatris-model-benchmark main.cpp)
target_link_libraries(squatris-model-benchmark PRIVATE
  squatris_backends
  squatris_core
  squatris_project_options
  nlohmann_json::nlohmann_json
)
set_target_properties(squatris-model-benchmark PROPERTIES
  OUTPUT_NAME "SquatrisModelBenchmark"
  FOLDER "Squatris/Tools"
)
squatris_stage_onnxruntime(squatris-model-benchmark)
