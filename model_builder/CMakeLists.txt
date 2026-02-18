cmake_minimum_required(VERSION 3.17)
project(build_engine LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

find_library(NVINFER_LIB nvinfer REQUIRED)
find_library(NVONNX_LIB nvonnxparser REQUIRED)

add_executable(build_engine build_engine.cpp)

target_link_libraries(build_engine 
    PRIVATE
    ${NVINFER_LIB}
    ${NVONNX_LIB}
    CUDA::cudart
)