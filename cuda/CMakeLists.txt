CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

PROJECT(CUDA-TEST LANGUAGES CXX CUDA)
ENABLE_LANGUAGE(CUDA) # 


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_75,code=sm_75; -std=c++11; -g;)



add_executable(test-stream
	stream.cu
)
