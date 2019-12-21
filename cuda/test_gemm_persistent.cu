#include <cstdio>
#include <assert.h>
#include <fstream>
#include <random>

#include <cuda.h>
#include <mma.h>
#include <cublas_v2.h>

#include "gemm_utils.h"
#include "gemm_wmma.cuh"
#include "test_utils.cuh"

using namespace nvcuda;
using std::cout;
using std::cin;
using std::endl;

#ifndef CPU_DEBUG
#define CPU_DEBUG 1
#endif

// xt_d_h * weights_c_matrix_d_h = buf_I_d_h
__host__ void matMultiplyOnCublas(
    half *weights_c_matrix_d_h,
    half *xt_d_h,
    half *buf_I_d_h,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{

  cublasStatus_t status;
  cublasHandle_t  handle;
  status = cublasCreate(&handle);
  checkCudaErrors(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaEvent_t start2, stop2;
  checkCudaErrors(cudaEventCreate(&start2));    
  checkCudaErrors(cudaEventCreate(&stop2));

  checkCudaErrors(cudaEventRecord(start2));
  checkCudaErrors(cublasGemmEx(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            M,N,K,
            &alpha,
            weights_c_matrix_d_h,
            CUDA_R_16F,
            lda,
            xt_d_h,
            CUDA_R_16F,
            ldb,
            &beta,
            buf_I_d_h,
            CUDA_R_16F,
            ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
            //CUBLAS_GEMM_DEFAULT
            ));

checkCudaErrors(cudaEventRecord(stop2));
  checkCudaErrors(cudaEventSynchronize(stop2));
  float milliseconds2 = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds2, start2, stop2));
  printf("cublas Time: %f ms\n", milliseconds2);

  cublasDestroy(handle);
  
}


// B * A = C
__host__ void gemm_cuBlas(
    bool transa, bool transb,
    half *A,
    half *B,
    half *C,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{

  cublasStatus_t status;
  cublasHandle_t  handle;
  status = cublasCreate(&handle);
  checkCudaErrors(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaEvent_t start2, stop2;
  checkCudaErrors(cudaEventCreate(&start2));    
  checkCudaErrors(cudaEventCreate(&stop2));
  checkCudaErrors(cudaEventRecord(start2));

  cublasOperation_t trans_a, trans_b;
  if(transa) trans_a = CUBLAS_OP_T;
  else trans_a = CUBLAS_OP_N;
  if(transb) trans_b = CUBLAS_OP_T;
  else trans_b = CUBLAS_OP_N;
  printf("M = %d, N = %d, N = %d \n", M, N, K);

  printf("lda = %d \n", lda);

  checkCudaErrors(cublasGemmEx(handle,
            //trans_a, trans_b,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M,  K,
            &alpha,
            B,
            CUDA_R_16F,
            K,
            A,
            CUDA_R_16F,
            K,
            &beta,
            C,
            CUDA_R_16F,
            N,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
            //CUBLAS_GEMM_DEFAULT
            ));

  checkCudaErrors(cudaEventRecord(stop2));
  checkCudaErrors(cudaEventSynchronize(stop2));
  float milliseconds2 = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds2, start2, stop2));
  printf("cublas Time: %f ms\n", milliseconds2);

  cublasDestroy(handle);
}



//
int call_gemm_wmma_shm_persistent(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm_persistent ...\n");

  int M, N, K, M_TILES, N_TILES, K_TILES;

  M = 16;  
  N = 16; 
  K = 16;

  int lda = K_GLOBAL + 16;
  int ldb = K_GLOBAL + 32;
  int ldc = N_GLOBAL + 64;

  M_TILES = M_GLOBAL / M;
  N_TILES = N_GLOBAL / N;
  K_TILES = K_GLOBAL / K;


  printf("M_GLOBAL: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N_GLOBAL: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K_GLOBAL: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  half *A_h = NULL;
  half *B_h = NULL;
  half *C_h = NULL;

#if CPU_DEBUG
  half *result_hD = NULL;
  half *result_host = NULL;
#endif

  A_h = (half*) malloc(sizeof(half) * M_GLOBAL * lda);
  B_h = (half*) malloc(sizeof(half) * N_GLOBAL * ldb);
  C_h = (half*) malloc(sizeof(half) * M_GLOBAL * ldc);

#if CPU_DEBUG
  result_hD   = (half*) malloc(sizeof(half) * M_GLOBAL * ldc);
  result_host = (half*) malloc(sizeof(half) * M_GLOBAL * ldc);
#endif

  half *A = NULL;
  half *B = NULL;
  half *C = NULL;

  checkCudaErrors(cudaMalloc((void**)&A, sizeof(half) * M_GLOBAL * lda));
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * ldb * N_GLOBAL));

  checkCudaErrors(cudaMalloc((void**)&C, sizeof(half) * M_GLOBAL * ldc));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);

  init_host_matrices_half_NT_align(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, 
      lda, ldb, ldc);
#if 0
  for(int i = 0; i < 512; i++){
    //printf("b_h[%d] = %f \n", i, __half2float(B_h[i]));
    printf("a[%d] = %f \n", i, __half2float(A_h[i]));
  }
  printf("Preparing data for GPU...\n");
#endif

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * lda, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * ldb, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(half) * M_GLOBAL * ldc, 
        cudaMemcpyHostToDevice));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  // If enough shared memory available on the GPU use high performant kernel
  // if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ)

  printf("Computing... gemm_wmma_shm_persistent \n");
  int runtimes = 100;
  //for(int i = 0; i < runtimes; i++)
    gemm_wmma_shm_persistent(false, false, 
        M_GLOBAL, N_GLOBAL, K_GLOBAL, 
        alpha, A, lda, 
        B, ldb, beta, C, ldc);

#if CPU_DEBUG
  checkCudaErrors(cudaMemcpy(result_hD, C, sizeof(half) * M_GLOBAL * ldc, 
        cudaMemcpyDeviceToHost));
#endif

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

#if 1
  memcpy(result_host, C_h, sizeof(half) * M_GLOBAL * ldc);
  matMultiplyOnHostHalf_NT_align(A_h, B_h, result_host,
      alpha, beta,
      M_GLOBAL, K_GLOBAL, lda,
      K_GLOBAL, N_GLOBAL, ldb,
      M_GLOBAL, N_GLOBAL, ldc);
#endif


  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      if (fabs(__half2float(result_hD[i*ldc+j]) -
            __half2float(result_host[i*ldc+j])) > 0.9f)
      {
        printf("mismatch i=%d, j=%d, result_hD=%f result_host=%f\n", i,j,
            __half2float(result_hD[i*ldc+ j]), 
            __half2float(result_host[i*ldc+ j]));
        //i = M_GLOBAL;
        //break;
      }
    }
  }

  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

#if 0
  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);
#endif

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));


  return 0;
}





void check_device()
{
  cudaSharedMemConfig shmCfg;
  checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  cudaDeviceGetSharedMemConfig(&shmCfg);
  if( shmCfg == cudaSharedMemBankSizeEightByte)
    printf("shm_bank_size = %d bytes \n", 8);
  else
    printf("shm_bank_size = %d bytes \n", 4);


}



int main(int argc, char **argv)
{

  // init.
  check_device();

  int M, N, K;
  int lda, ldb, ldc;
  M = 128; N = 3072;  K =  1024;

  //test_gemm_wmma_shm(M, N, K);
  //test_gemm_wmma_shm_half(M, N, K); // ok
  //call_gemm_wmma_shm_half(M, N, K); // ok

#if 1
  M = 128; N = 3072; K = 1024;
  lda = K+32, ldb = N+64, ldc = N+128;
  for(int i =0 ; i < 1; i++){
    M = 128;
    //call_gemm_wmma_shm_half(M, N, K); // ok

    //M = 32; N =6144; K = 16;
    //M = 128; N = 512; K = 512;
    //call_gemm_wmma_shm_half_config(M, N, K);

    //M = 128; N = 3072; K = 1024;
    M = 128; N = 512; K = 512;
    //M = 128; N = 256; K = 512;
    //M = 32; N =6144; K = 16;
    //call_gemm_wmma_shm_half_128_16(M, N, K);
    call_gemm_wmma_shm_persistent(M, N, K);
    printf("Finish M = %d \n", M);
    M += 16;
  }
#endif

  //test_gemm_wmma_shm_NN(M, N, K);
  //test_gemm_wmma(M, N, K );
  //test_gemm_wmma_half(M, N, K);

  //call_gemm_wmma_half(M, N, K, lda, ldb, ldc);
  return 0;
}
