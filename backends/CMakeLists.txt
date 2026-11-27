add_library(squatris_backends STATIC
  camera/opencv_camera_source.cpp
  camera/prerecorded_source.cpp
  camera/sequential_frame_source.cpp
  camera/synthetic_source.cpp
  inference/mock_pose_estimator.cpp
  rtmpose/model_math.cpp
  rtmpose/rtmpose_estimator.cpp
)
target_include_directories(squatris_backends PUBLIC include)
target_link_libraries(squatris_backends
  PUBLIC squatris_core squatris_project_options
  PRIVATE opencv_core opencv_imgproc opencv_videoio
)
target_compile_definitions(squatris_backends PUBLIC
  SQUATRIS_HAS_ONNXRUNTIME=$<BOOL:${SQUATRIS_ENABLE_ONNXRUNTIME}>
  SQUATRIS_ORT_HAS_CUDA=$<BOOL:${SQUATRIS_ORT_HAS_CUDA}>
  SQUATRIS_ONNXRUNTIME_VERSION="${SQUATRIS_ONNXRUNTIME_VERSION}"
)
if(SQUATRIS_ENABLE_ONNXRUNTIME)
  target_include_directories(squatris_backends PRIVATE "${SQUATRIS_ONNXRUNTIME_ROOT}/include")
  target_link_libraries(squatris_backends PRIVATE "${SQUATRIS_ONNXRUNTIME_LIBRARY}")
endif()
set_target_properties(squatris_backends PROPERTIES FOLDER "Squatris")
