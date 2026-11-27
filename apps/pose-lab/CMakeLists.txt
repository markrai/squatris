add_executable(squatris-pose-lab main.cpp)
target_link_libraries(squatris-pose-lab PRIVATE
  squatris_backends
  squatris_core
  squatris_project_options
  opencv_core
  opencv_highgui
  opencv_imgproc
)
set_target_properties(squatris-pose-lab PROPERTIES
  OUTPUT_NAME "SquatrisPoseLab"
  FOLDER "Squatris/Apps"
)
squatris_stage_onnxruntime(squatris-pose-lab)
