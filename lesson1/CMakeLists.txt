cmake_minimum_required(VERSION 3.0)
project(lesson1)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86")


set(CUDNN_DIR    "/root/dep/cudnn-linux-x86_64-8.9.3.28_cuda11-archive")
set(TENSORRT_DIR "/root/dep/TensorRT-8.4.1.5")
set(PROTOBUF_DIR "/root/dep/protobuf-v3.11.4") # source install 3.11.4

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

message("Usage Protobuf ${PROTOBUF_DIR}")
message("Usage CUDA ${CUDA_TOOLKIT_ROOT_DIR}")

include_directories(../src/tensorRT)


include_directories(    
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${PROTOBUF_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)

link_directories(
    ${PROTOBUF_DIR}/lib
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDNN_DIR}/lib
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

file(GLOB_RECURSE cpp_srcs ../src/tensorRT/*.cpp)

add_executable(${PROJECT_NAME} import_trt.cc ${cpp_srcs})

target_link_libraries(${PROJECT_NAME} nvinfer nvinfer_plugin)
target_link_libraries(${PROJECT_NAME} cuda cublas cudart cudnn)
target_link_libraries(${PROJECT_NAME} protobuf pthread)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})




