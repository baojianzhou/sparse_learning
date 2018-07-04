cmake_minimum_required(VERSION 3.10)
project(sparse_learning C)

set(CMAKE_C_STANDARD 11)
# ---------------------- for python3.6 -----------------------------
find_package(PythonLibs REQUIRED)
set(PYTHON_EXECUTABLE "/usr/bin/python3.6")
set(PYTHON_INCLUDE_DIR "/usr/include/python3.6")
set(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.6m.so")
include_directories(${PYTHON_INCLUDE_DIRS})
# ------------------------------------------------------------------
add_executable(test_logistic_loss src/test_loss_func.c src/loss_func.h src/loss_func.c)
target_link_libraries(test_logistic_loss -lopenblas -lpthread -lgfortran -lm -O2)

add_executable(test_logistic_regression src/logistic_regression.c src/logistic_regression.h src/test_logistic_regression.c)
target_link_libraries(test_logistic_regression -lopenblas -lpthread -lgfortran -lm -O2)

add_executable(test_python_wrapper src/python_wrapper.c)
target_link_libraries(test_python_wrapper ${PYTHON_LIBRARIES} -lopenblas -lpthread -lgfortran -lm -O2)