find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/ops)
add_library(cuda ops/vector.cu)
target_link_libraries(cuda ${CUDA_LIBRARIES})
