add_library(squatris_core STATIC
  inference/tracking_assessment.cpp
  pose/geometry.cpp
  pose/mapping.cpp
  pose/pose.cpp
  metrics/latency_statistics.cpp
)
target_include_directories(squatris_core PUBLIC include)
target_link_libraries(squatris_core PUBLIC squatris_project_options)
set_target_properties(squatris_core PROPERTIES FOLDER "Squatris")
