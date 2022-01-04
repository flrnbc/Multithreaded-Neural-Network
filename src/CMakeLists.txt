# libraries
# need STATIC, SHARED...?
add_library(LibFunction function.cpp)
add_library(LibLossFunction loss_function.cpp)
add_library(LibTransformation transformation.cpp)
add_library(LibLayerCache layer_cache.cpp)
add_library(LibLayer layer.cpp)
add_library(LibSequentialNN sequential_nn.cpp)

# include paths TODO: how to streamline?
target_include_directories(LibLossFunction PUBLIC
                            "${PROJECT_BINARY_DIR}" # necessary?
                            "${PROJECT_SOURCE_DIR}/eigen"
)
target_include_directories(LibTransformation PUBLIC
                            "${PROJECT_BINARY_DIR}" # necessary?
                            "${PROJECT_SOURCE_DIR}/eigen"
)
target_include_directories(LibLayerCache PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/eigen"
)
target_include_directories(LibLayer PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/eigen"
)
target_include_directories(LibSequentialNN PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/eigen"
)
