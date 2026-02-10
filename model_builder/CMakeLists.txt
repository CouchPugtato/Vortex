cmake_minimum_required(VERSION 3.10)
project(build_engine)

find_package(CUDA REQUIRED)

# TensorRT Paths (Adjust if needed, usually in /usr/lib/aarch64-linux-gnu/ on Jetson)
# We assume standard JetPack install
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

add_executable(build_engine build_engine.cpp)

# Link against TensorRT and NVONNXParser
target_link_libraries(build_engine 
    nvinfer 
    nvonnxparser 
    ${CUDA_LIBRARIES}
)
