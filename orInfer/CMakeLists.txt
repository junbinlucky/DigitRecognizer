cmake_minimum_required(VERSION 3.16)

project(onnx_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")

set(ONNXRUNTIME_ROOT_DIR "/usr/local")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so")

add_executable(onnx_infer onnx_infer.cpp)

target_include_directories(onnx_infer
    PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(onnx_infer
    PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
    pthread dl
)

