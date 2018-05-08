# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(saliency)

find_package(OpenCV REQUIRED)
#C++ 11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -O3 -fopenmp")
# If the package has been found, several variables will

#生成文件放在bin目录下
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

#生成库
add_library(slic ${CMAKE_CURRENT_SOURCE_DIR}/source/SLIC.cpp)
add_library(rgb2hsi ${CMAKE_CURRENT_SOURCE_DIR}/source/RGB2HSI.cpp)
#cpp放在source文件下
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
#h文件
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# Declare the executable target built from your sources

add_executable(main main.cpp )
# Link your application with OpenCV libraries

#target_link_libraries(saliency PUBLIC ${OpenCV_LIBS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main slic rgb2hsi)

