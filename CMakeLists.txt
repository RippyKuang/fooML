cmake_minimum_required(VERSION 3.10)
project(fooml VERSION 1.0 LANGUAGES CXX CUDA)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS}  -Wall")

add_subdirectory(autograd)
add_subdirectory(tensor)
add_executable(fooml main.cpp)

target_link_libraries(fooml PUBLIC ag cuda)
target_include_directories(fooml PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/autograd/inc"
                          "${PROJECT_SOURCE_DIR}/tensor"
                          )