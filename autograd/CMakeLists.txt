include_directories("${PROJECT_BINARY_DIR}"
		            "${CMAKE_CURRENT_SOURCE_DIR}/inc"
					"${PROJECT_SOURCE_DIR}/tensor"
		            )
add_library(ag STATIC autograd.cpp)