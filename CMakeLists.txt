cmake_minimum_required(VERSION 3.10)
project(IntersectionEnforcement)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_INCLUDE_DIR "C:/onnxruntime/include")
set(ONNXRUNTIME_LIB_DIR "C:/onnxruntime/lib")

include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

add_executable(enforcement_camera "CW&TL_logic.cpp")

target_link_libraries(enforcement_camera ${OpenCV_LIBS} onnxruntime)