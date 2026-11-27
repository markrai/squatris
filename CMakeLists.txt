cmake_minimum_required(VERSION 3.28)
project(Squatris VERSION 0.1.0 LANGUAGES CXX)

option(SQUATRIS_BUILD_APPS "Build Pose Lab" ON)
option(SQUATRIS_BUILD_BENCHMARKS "Build benchmark tools" ON)
option(SQUATRIS_BUILD_TESTS "Build automated tests" ON)
option(SQUATRIS_ENABLE_ONNXRUNTIME "Build the RTMPose ONNX Runtime backend" OFF)
option(SQUATRIS_ENABLE_HARDWARE_TESTS "Register explicit CUDA and camera qualification tests" OFF)
option(SQUATRIS_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)
set(SQUATRIS_ONNXRUNTIME_ROOT "" CACHE PATH "Extracted ONNX Runtime package root")
set(SQUATRIS_ONNXRUNTIME_VERSION "1.20.1")
set(SQUATRIS_HARDWARE_VIDEO "" CACHE FILEPATH "Qualification video for hardware tests")
set(SQUATRIS_HARDWARE_PROVIDER "cuda" CACHE STRING "Provider used by hardware tests")
set_property(CACHE SQUATRIS_HARDWARE_PROVIDER PROPERTY STRINGS cpu cuda)

find_package(OpenCV CONFIG REQUIRED COMPONENTS core highgui imgproc videoio)
find_package(nlohmann_json CONFIG REQUIRED)

add_library(squatris_project_options INTERFACE)
target_compile_features(squatris_project_options INTERFACE cxx_std_20)
if(MSVC)
  target_compile_options(squatris_project_options INTERFACE /W4 /permissive- /EHsc /utf-8)
  if(SQUATRIS_WARNINGS_AS_ERRORS)
    target_compile_options(squatris_project_options INTERFACE /WX)
  endif()
else()
  target_compile_options(squatris_project_options INTERFACE -Wall -Wextra -Wpedantic)
  if(SQUATRIS_WARNINGS_AS_ERRORS)
    target_compile_options(squatris_project_options INTERFACE -Werror)
  endif()
endif()

if(SQUATRIS_ENABLE_ONNXRUNTIME)
  if(NOT EXISTS "${SQUATRIS_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "Set SQUATRIS_ONNXRUNTIME_ROOT to an extracted ONNX Runtime 1.20.1 package")
  endif()
  find_library(SQUATRIS_ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS "${SQUATRIS_ONNXRUNTIME_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
  file(GLOB SQUATRIS_ONNXRUNTIME_DLLS "${SQUATRIS_ONNXRUNTIME_ROOT}/lib/*.dll")
  if(EXISTS "${SQUATRIS_ONNXRUNTIME_ROOT}/include/cuda_provider_factory.h")
    set(SQUATRIS_ORT_HAS_CUDA ON)
  else()
    set(SQUATRIS_ORT_HAS_CUDA OFF)
  endif()
endif()

function(squatris_stage_onnxruntime target)
  if(SQUATRIS_ENABLE_ONNXRUNTIME AND WIN32)
    foreach(SQUATRIS_ORT_DLL IN LISTS SQUATRIS_ONNXRUNTIME_DLLS)
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SQUATRIS_ORT_DLL}"
          "$<TARGET_FILE_DIR:${target}>"
      )
    endforeach()
  endif()
endfunction()

add_subdirectory(core)
add_subdirectory(backends)

if(SQUATRIS_BUILD_APPS)
  add_subdirectory(apps/pose-lab)
endif()

if(SQUATRIS_BUILD_BENCHMARKS)
  add_subdirectory(tools/benchmark)
endif()

include(CTest)
if(BUILD_TESTING AND SQUATRIS_BUILD_TESTS)
  add_subdirectory(tests)
endif()
