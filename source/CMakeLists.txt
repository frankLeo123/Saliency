set(SOURCE_FILES SLIC.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -O3 -fopenmp")

add_library(saliency ${SOURCE_FILES})
target_link_libraries(saliency PUBLIC ${OpenCV_LIBS})
#set_target_properties(chessboard PROPERTIES RUNTIME_OUTPUT_DIRECTORY> "${PROJECT_SOURCE_DIR}/build")
