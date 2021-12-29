cmake_minimum_required(VERSION 3.7)

# set project for tests
project (TEST)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# source subdirectory
add_subdirectory(src)
add_subdirectory(tests)

add_executable(TestFunctions tests/test_function.cpp)

target_link_libraries(TestFunctions PUBLIC Function TestFunction)

target_include_directories(TestFunction PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/src"
                            "${PROJECT_SOURCE_DIR}/tests"
)


# add_library(Transformation transformation.cpp transformation.h)
# add_library(LayerCache layer_cache.cpp layer_cache.h)
# add_library(Layer layer.cpp layer.h)
# add_library(SequentialNN sequential_nn.cpp sequential_nn.h)

# link libraries
# target_link_libraries(Transformation Function) # Transformation needs Function
# target_link_libraries(Layer LayerCache Transformation)
# target_link_libraries(SequentialNN Layer)

#add_executable(demo main.cpp)
#target_link_libraries(demo Function)