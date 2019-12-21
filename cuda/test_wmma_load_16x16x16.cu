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
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  // Initialize the output to zero
  wmma::fill_fragment(c_frag, 0.0f);
  // Load the inputs
  wmma::load_matrix_sync(a_frag, a, 16);
  wmma::load_matrix_sync(b_frag, b, 16);
  // Perform the matrix multiplication
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  // Store the output
  wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
  //printf("the c[%d]:%f\n",tid, c[tid]);
}
int main()
{
   half  *A, *B;
   float *C;
   half *dev_A, *dev_B; 
   float  *dev_C;
   const int size_mat = 16*16;
   A = (half*)malloc(sizeof(half)*size_mat);
   B = (half*)malloc(sizeof(half)*size_mat);
   C = (float*)malloc(sizeof(float)*size_mat);
   cudaMalloc((void**)&dev_A, sizeof(half)*size_mat);
   cudaMalloc((void**)&dev_B, sizeof(half)*size_mat);
   cudaMalloc((void**)&dev_C, sizeof(float)*size_mat);
   for(int i=0; i<size_mat; ++i)
   {
      A[i] =  __float2half(1.0f);
      B[i] = __float2half(1.0f);
      C[i] = 0;
   }
   cudaMemcpy(dev_A, A, sizeof(half)*size_mat, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_B, B, sizeof(half)*size_mat, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_C, C, sizeof(half)*size_mat, cudaMemcpyHostToDevice);
   dim3 grid(2,2);
   dim3 block(2,4,4);
   wmma_ker<<<grid, block>>>(dev_A, dev_B, dev_C);
  
   cudaMemcpy(C, dev_C, sizeof(float)*size_mat, cudaMemcpyDeviceToHost);
   for(int i=0; i<size_mat; ++i)
   {
     if(fabs(C[i] -16) > 0.1)
     {
      printf("the C[%d]:%f\n",i,C[i]);
     }
   }
  return 0;
}
