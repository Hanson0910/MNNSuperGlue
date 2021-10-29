cmake_minimum_required(VERSION 3.0)
project(superglue)

find_package(OpenCV 3.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#这里链接的时编译后的ncnn的include和lib文件夹，根据自己的路径去更改
include_directories(yourMNNPath/MNN-master/build/install/usr/local/include)
link_directories(yourMNNPath/MNN-master/build/install/usr/local/lib)
link_directories(yourMNNPath/MNN-master/build/source/backend/cuda)

include_directories(./inc)
aux_source_directory(./src DIR_SRCS)
message(WARNING,${DIR_SRCS})

set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
message(WARNING,${CMAKE_BUILD_TYPE})

add_executable(kptsdet main.cpp ${DIR_SRCS})
target_link_libraries(kptsdet ${OpenCV_LIBS} libMNN.so libMNN_Cuda_Main.so)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/out)
add_library(SuperGlue SHARED ${DIR_SRCS})
