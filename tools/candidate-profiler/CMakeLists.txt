add_executable(squatris-candidate-profiler main.cpp)
target_link_libraries(squatris-candidate-profiler PRIVATE
  squatris_backends
  squatris_core
  squatris_project_options
  nlohmann_json::nlohmann_json
)
set_target_properties(squatris-candidate-profiler PROPERTIES
  OUTPUT_NAME "SquatrisCandidateProfiler"
  FOLDER "Squatris/Tools"
)
squatris_stage_onnxruntime(squatris-candidate-profiler)
