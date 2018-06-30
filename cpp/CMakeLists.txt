cmake_minimum_required(VERSION 2.8)
project(sparse_proj)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpermissive -std=c++11 -lpython2.7")
# Find necessary packages
find_package(PythonLibs 2.7 REQUIRED)
message(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})


add_executable(test_fast_pcst test_fast_pcst.cpp pcst_fast.h)
add_executable(test_head_tail test_head_tail.cpp head_tail.h)
add_executable(test_main_wrapper main_wrapper.cpp head_tail.h)
add_executable(test_union_find test_union_find.cpp union_find.h)
add_executable(test_loss_func test_loss_func.c loss_func.c loss_func.h)

add_library(proj_algo SHARED head_tail.h pcst_fast.h main_wrapper.cpp)
target_link_libraries(proj_algo python2.7)
target_link_libraries(test_loss_func -lm)