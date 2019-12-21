#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
using namespace cooperative_groups;
#if 0
__global__  void Add(float*  A, float* B, float* C, int size_array)
{
  grid_group  g = this_grid();
  size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
  if(tid < size_array )
  {
    A[tid]  = 1.0;
    B[tid]  = 2.0;
    C[tid/2] = A[tid] + B[tid];
  }
  //g.sync();
  if(tid < size_array )
  {
    C[tid] = A[tid] + B[tid] + C[tid/2];
  }
}
#endif

__device__  void Add1(float*  A, float* B, float* C, int size_array)
{
  //grid_group  g = this_grid();
  size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
  if(tid < size_array )
  {
    A[tid]  = 1.0;
    B[tid]  = 2.0;
    C[tid/2] = A[tid] + B[tid];
  }
}
  //g.sync();
__device__  void Add2(float*  A, float* B, float* C, int size_array)
{
  size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
  if(tid < size_array )
  {
    C[tid] = A[tid] + B[tid] + C[tid/2];
  }
}

__global__  void Add(float*  A, float* B, float* C, int size_array)
{
  grid_group  g = this_grid();
  Add1(A,B,C,size_array);
  //g.sync();
  Add2(A,B,C,size_array);
}

__global__  void Add_plus(float*  A, float* B, float* C, int size_array)
{
  size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
  if(tid < size_array )
  {
    C[tid] += A[tid] + B[tid];
  }
}

int main()
{
  float   *A, *B, *C;
  const size_t  Array_size = 128;
  cudaMalloc((void**)&A, sizeof(float)*Array_size);
  cudaMalloc((void**)&B, sizeof(float)*Array_size);
  cudaMalloc((void**)&C, sizeof(float)*Array_size);
  Add<<<16,13>>>(A, B, C,Array_size);
 // Add_plus<<<16,8>>>(A, B, C,Array_size);
  float   *h_C;
  h_C = (float*)malloc(sizeof(float)*Array_size);
  cudaMemcpy(h_C, C, sizeof(float)*Array_size, cudaMemcpyDeviceToHost);
  for(int i = 0; i< Array_size; ++i)
  {
    if(fabs(h_C[i] - 6.0) > 0.1) 
    {
      printf("the C[%d]:%f\n",i,h_C[i]);
    }
  }

  
  return 0;
}
