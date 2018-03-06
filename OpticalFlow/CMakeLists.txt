cmake_minimum_required(VERSION 2.8)
project( lk )

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg -std=c++11")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( lk opticalflow.cpp )
target_link_libraries( lk ${OpenCV_LIBS} )
