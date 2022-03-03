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
add_executable(TestLossFunction tests/test_loss_function.cpp)
add_executable(TestTransformation tests/test_transformation.cpp)
add_executable(TestLayerCache tests/test_layer_cache.cpp)
add_executable(TestLayer tests/test_layer.cpp)
add_executable(TestSequentialNN tests/test_sequential_nn.cpp)
add_executable(TestOptimizer tests/test_optimizer.cpp)
add_executable(TestDataParser tests/test_data_parser.cpp)
add_executable(Main main.cpp)

# linking libraries
target_link_libraries(TestFunction PUBLIC LibFunction)
target_link_libraries(TestLossFunction PUBLIC LibLossFunction)
target_link_libraries(TestTransformation PUBLIC LibFunction LibTransformation)
target_link_libraries(TestLayerCache PUBLIC LibLayerCache)
target_link_libraries(TestLayer PUBLIC LibFunction LibTransformation LibLayerCache LibLayer LibLossFunction)
target_link_libraries(TestSequentialNN PUBLIC LibFunction LibLossFunction LibTransformation LibLayer LibLayerCache LibSequentialNN)
target_link_libraries(TestOptimizer PUBLIC LibOptimizer LibFunction LibLossFunction LibTransformation LibLayer LibLayerCache LibSequentialNN)
target_link_libraries(TestDataParser PUBLIC LibDataParser)
target_link_libraries(Main PUBLIC LibOptimizer LibFunction LibLossFunction LibTransformation LibLayer LibLayerCache LibSequentialNN LibDataParser)