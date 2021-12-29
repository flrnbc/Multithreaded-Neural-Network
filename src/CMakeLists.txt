# need STATIC, SHARED...?
add_library(LibFunction function.cpp)
add_library(LibTransformation transformation.cpp)
#add_library(LibLayer layer.cpp layer.h layer_cache.cpp layer_cache.h transformation.cpp transformation.h)
target_include_directories(LibTransformation PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/eigen"
)