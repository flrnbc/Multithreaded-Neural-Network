add_library(LibTestFunction test_function.cpp)
add_library(LibTestTransformation test_transformation.cpp)
target_include_directories(LibTestTransformation PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/eigen"
)

