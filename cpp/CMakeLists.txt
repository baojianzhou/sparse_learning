cmake_minimum_required(VERSION 2.8)
project(sparse_proj)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpermissive -std=c++11 -lpython2.7 -ffast-math")
set(OPENBLAS_INCLUDE_DIR "/network/rit/lab/ceashpc/bz383376/opt/openblas-0.3.1/include")
set(OPENBLAS_INCLUDE "/network/rit/lab/ceashpc/bz383376/opt/openblas-0.3.1/include")
set(OPENBLAS_LIB "/network/rit/lab/ceashpc/bz383376/opt/openblas-0.3.1/lib")
# Find necessary packages
find_package(PythonLibs 2.7 REQUIRED)
message(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${OPENBLAS_INCLUDE_DIR})

add_executable(test_fast_pcst pcst_fast_test.cpp pcst_fast.h)
add_executable(test_head_tail head_tail_test.cpp head_tail.h)
add_executable(test_main_wrapper main_wrapper.cpp head_tail.h pcst_fast.h ghtp_algo.h)
add_executable(test_union_find union_find_test.cpp union_find.h)
add_executable(test_ghtp_algo ghtp_algo_test.cpp ghtp_algo.h)

add_library(proj_module SHARED head_tail.h pcst_fast.h main_wrapper.cpp union_find.h ghtp_algo.h)
<<<<<<< HEAD

target_link_libraries(test_main_wrapper -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lm -lpython2.7 -lpthread)
target_link_libraries(proj_module python2.7 -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lm -lpython2.7 -lpthread)
target_link_libraries(test_ghtp_algo -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lm -lpthread)

=======
target_link_libraries(test_main_wrapper python2.7 -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)
target_link_libraries(proj_module python2.7 -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lm -lpython2.7 -lopenblas -lpthread)
target_link_libraries(test_ghtp_algo -I${OPENBLAS_INCLUDE} -L${OPENBLAS_LIB} -lm -lopenblas -lpthread)
>>>>>>> 099a5bc7613c31addae2324a6fed960147f03849
