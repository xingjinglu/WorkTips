#include <mma.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <stdio.h>
using namespace nvcuda;
using namespace std;

__global__ void wmma_ker(half *a, half *b, float *c) {
  // Declare the fragments
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  //float f_a = __half2float(a[tid]);
  //float f_b = __half2float(b[tid]);
  //printf("the a[%d]:%f\n",tid, f_a);
  //printf("the b[%d]:%f\n",tid, f_b);
  wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 8, 32, 16, float> c_frag;
  // Initialize the output to zero
  wmma::fill_fragment(c_frag, 0.0f);
  // Load the inputs
  wmma::load_matrix_sync(a_frag, a, 8);
  wmma::load_matrix_sync(b_frag, b, 32);
  // Perform the matrix multiplication
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  // Store the output
  wmma::store_matrix_sync(c, c_frag, 32, wmma::mem_row_major);
  //printf("the c[%d]:%f\n",tid, c[tid]);
}
int main()
{
   half  *A, *B;
   float *C;
   half *dev_A, *dev_B; 
   float  *dev_C;
   const int size_mat_A = 8*16;
   const int size_mat_B = 16*32;
   const int size_mat_C = 8*32;
   A = (half*)malloc(sizeof(half)*size_mat_A);
   B = (half*)malloc(sizeof(half)*size_mat_B);
   C = (float*)malloc(sizeof(float)*size_mat_C);
   cudaMalloc((void**)&dev_A, sizeof(half)*size_mat_A);
   cudaMalloc((void**)&dev_B, sizeof(half)*size_mat_B);
   cudaMalloc((void**)&dev_C, sizeof(float)*size_mat_C);
   for(int i=0; i<size_mat_A; ++i)
   {
      A[i] =  __float2half(2.0f);
   }
   for(int i=0; i<size_mat_B; ++i)
   {
      B[i] =  __float2half(1.0f);
   }
   for(int i=0; i<size_mat_B; ++i)
   {
      C[i] =  0;
   }
   cudaMemcpy(dev_A, A, sizeof(half)*size_mat_A, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_B, B, sizeof(half)*size_mat_B, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_C, C, sizeof(float)*size_mat_C, cudaMemcpyHostToDevice);
   dim3 grid(1,1);
   dim3 block(32,1,1);
   printf("wmma_ker\n");
   wmma_ker<<<grid, block>>>(dev_A, dev_B, dev_C);
   printf("wmma_ker\n");
  
   cudaMemcpy(C, dev_C, sizeof(float)*size_mat_C, cudaMemcpyDeviceToHost);
   for(int i=0; i<size_mat_C; ++i)
   {
     if(fabs(C[i] -16) > 0.1)
     {
      printf("the C[%d]:%f\n",i,C[i]);
     }
   }
  return 0;
}
