cmake_minimum_required(VERSION 2.8)
project(head_tail_proj)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpermissive -std=c++11")
# Find necessary packages
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
# Build our library
add_library(head_tail SHARED pcst_fast.cc pcst_fast.h head_tail.h head_tail.cpp )
target_link_libraries(head_tail ${Boost_LIBRARIES})
# don't prepend wrapper library name with lib
set_target_properties(head_tail PROPERTIES PREFIX "")
include_directories("./")
add_executable(test_main pcst_fast.h pcst_fast.cc main.cpp )