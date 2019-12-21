#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "helper_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h> // Occupancy calculator in the CUDA toolkit

// Device code 
__global__ void MyKernelEx1(int *d, int *a, int *b) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  d[idx] = a[idx] * b[idx];
}

__global__ void MyKernelEx2(int *array, int arrayCount) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < arrayCount) { 
    array[idx] *= array[idx];
  } 
}

// Host code

int launchMyKernel(int *array, int arrayCount) {
  int blockSize;  // The launch configurator returned block size

  int minGridSize; // The minimum grid size needed to achieve the 
  // maximum occupancy for a full device launch
  int gridSize;  // The actual grid size needed, based on input size

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
      (void*)MyKernelEx2, 0,arrayCount); // Round up according to
  array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;
  MyKernelEx2<<<gridSize, blockSize>>>(array, arrayCount);
  cudaDeviceSynchronize();
  return 0;
  // If interested, the occupancy can be calculated with //
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor
} 

int main() {
  int numBlocks; // Occupancy in terms of active blocks
  int blockSize = 64;

  // These variables are used to convert occupancy to warps 
  int device;
  cudaDeviceProp prop; 
  int activeWarps; 
  int maxWarps;

  cudaGetDevice(&device); 
  cudaGetDeviceProperties(&prop, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
      MyKernelEx1, blockSize, 0);

  activeWarps = numBlocks * blockSize / prop.warpSize;
  maxWarps = prop.maxThreadsPerMultiProcessor /
    prop.warpSize;
  std::cout << "Num of blocks: " << numBlocks <<
    std::endl;
  //  std::cout << "Max num of warps per SM: " <<
  //  maxWarps << std::endl;
  //  std::cout << "Max threads per SM: " <<
  //  prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Occupancy: " << (double)activeWarps
    / maxWarps * 100 << "%" << std::endl;
  return 0;
} 
