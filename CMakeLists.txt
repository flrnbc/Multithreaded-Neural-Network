cmake_minimum_required(VERSION 3.7)

# set project for tests
project (TEST)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# source subdirectory 
# TODO: Definitely needed but why? Difference to target_include_directories?
add_subdirectory(src)
add_subdirectory(tests)

# executables
add_executable(TestFunction tests/test_function.cpp)
add_executable(TestTransformation tests/test_transformation.cpp)

# linking libraries
target_link_libraries(TestFunction PUBLIC LibFunction LibTestFunction)
target_link_libraries(TestTransformation PUBLIC LibFunction LibTransformation LibTestTransformation)

# include directories for above targets
target_include_directories(TestFunction PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/src"
                            "${PROJECT_SOURCE_DIR}/tests"
)
target_include_directories(TestTransformation PUBLIC
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