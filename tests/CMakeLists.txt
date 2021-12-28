cmake_minimum_required(VERSION 3.7)

# set project
project(UdacityCapstoneCppTests VERSION 0.1)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(${PARENT_DIR}/eigen)

add_executable(test_sequential_nn tests_sequential_nn.cpp ../src/sequential_nn.cpp ../src/layer.cpp ../src/layer_cache.cpp ../src/transformation.cpp ../src/function.cpp)