# libraries
# need STATIC, SHARED...?
add_library(LibFunction function.cpp function.h)
add_library(LibLossFunction loss_function.cpp loss_function.h)
add_library(LibTransformation transformation.cpp transformation.h)
add_library(LibLayerCache layer_cache.cpp layer_cache.h)
add_library(LibLayer layer.cpp layer.h)
add_library(LibSequentialNN sequential_nn.cpp sequential_nn.h)
add_library(LibOptimizer optimizer.cpp optimizer.h)
add_library(LibDataParser data_parser.cpp data_parser.h)

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
target_include_directories(LibOptimizer PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/eigen"
)
target_include_directories(LibDataParser PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/eigen"
)       