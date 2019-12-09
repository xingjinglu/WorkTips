#include <cstdio>
#include <assert.h>

#include <cuda.h>
#include <mma.h>

#include <cublas_v2.h>
#include <random>
#include <fstream>


#include "gemm_utils.h"
#include "gemm_wmma.cuh"

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

__host__ void init_host_matrices_NN(half *a, half *b, float *c, int M_GLOBAL, 
    int N_GLOBAL, int K_GLOBAL)
{
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i*K_GLOBAL+j] = __float2half(rand() % 3);
    }
  }

  for (int i = 0; i < K_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      b[i*N_GLOBAL+j] = __float2half(rand() % 3);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] =  (float)(rand() % 3);
  }
}

__host__ void init_host_matrices(half *a, half *b, float *c, int M_GLOBAL, 
    int N_GLOBAL, int K_GLOBAL)
{
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i*K_GLOBAL+j] = __float2half(rand() % 3);
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i*K_GLOBAL+j] = __float2half(rand() % 3);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] =  (float)(rand() % 3);
  }
}
__host__ void init_host_matrices_half_align_file(half *a, half *b, half *c, int
    M_GLOBAL, int N_GLOBAL, int K_GLOBAL,int lda, int ldb, int ldc, char*
    fileNameA , char* fileNameB)
{
  //std::ifstream a("1/weights_c_h_480_0.check");
  //std::ifstream b("1/xt_h_2_480_0.check");
  std::ifstream fileA(fileNameA);
  std::ifstream fileB(fileNameB);
  float a_tmp;
  float b_tmp;

  for (int i =0; i< M_GLOBAL; ++i)
  {
    for(int j=0; j< K_GLOBAL; ++j)
    {
      fileA >> a_tmp;
      a[i*lda +j] = __float2half(a_tmp);


    }

  }
  for (int i =0; i< K_GLOBAL; ++i)
  {
    for(int j=0; j< N_GLOBAL; ++j)
    {
      fileB >> b_tmp;
      b[i*ldb+j] = __float2half(b_tmp);
      //printf("the xt_h is:%f\n", b[i*ldb +j]);
    }
  }
}
__host__ void init_host_matrices_half_align(half *a, half *b, half *c, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,int lda, int ldb, int ldc)
{

  std::default_random_engine random(time(NULL));
  std::uniform_real_distribution<float> dis2(0.0, 1.0);

  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      //a[i*lda+j] = __float2half((rand()) % 117/17);
      a[i*lda+j] = __float2half(dis2(random));
    }
  }

  for (int i = 0; i < K_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      //b[i*ldb+j] = __float2half((rand() + i + j) % 117/17);
      b[i*ldb+j] = __float2half(dis2(random));
    }
  }
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      //c[i*ldc+j] = __float2half((rand() + i*j) % 3);
      c[i*ldc+j] = __float2half(dis2(random));
    }
  }

}

__host__ void init_host_matrices_half(half *a, half *b, half *c, int M_GLOBAL, 
    int N_GLOBAL, int K_GLOBAL)
{
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i*K_GLOBAL+j] = __float2half((rand()) % 3);
      //printf("%f \n", (float)((half)(float)(rand() % 3)));
      //printf("%f \n", (float)((rand() % 3)));
    }
  }

  for (int i = 0; i < K_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      b[i*N_GLOBAL+j] = __float2half((rand() + i + j) % 3);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] =  __float2half((rand() + t) % 3);
  }
}
__host__ void init_host_matrices_half_NT_align(half *a, half *b, half *c, int M_GLOBAL, 
    int N_GLOBAL, int K_GLOBAL, int lda, int ldb, int ldc)
{

  std::default_random_engine random(time(NULL));
  std::uniform_real_distribution<float> dis2(0.0, 1.0);
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      //a[i*lda+j] = __float2half((rand()) % 3);
      //a[i*lda+j] = __float2half(1.0f);
      a[i*lda+j] = __float2half(dis2(random));
      //a[i*K_GLOBAL+j] = __float2half(1.0f);
      //printf("%f \n", (float)((half)(float)(rand() % 3)));
      //printf("%f \n", (float)((rand() % 3)));
    }
  }

#if 0
  for(int j = 0; j < 16; j++){
    printf("host_a[0][%d] = %f\n", j, __half2float(a[0 * lda + j]));
    printf("host_a[16][%d] = %f\n", j, __half2float(a[16 * lda + j]));
  }
#endif

#if 1
  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      //b[i*K_GLOBAL+j] = __float2half((rand() + i + j) % 3);
      //b[i*ldb+j] = __float2half(1.0f);
      b[i*ldb+j] = __float2half(dis2(random));
    }
  }
#endif
#if 0
  for (int i = 0; i < K_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      //b[i*K_GLOBAL+j] = __float2half((rand() + i + j) % 3);
      b[i*ldb+j] = __float2half(1.0f);
    }
  }
#endif
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      //c[i*ldc+j] = __float2half((rand() + i*j) % 3);
      c[i*ldc+j] = __float2half(dis2(random));
      //printf("%f \n", (float)((half)(float)(rand() % 3)));
      //printf("%f \n", (float)((rand() % 3)));
    }
  }

}

__host__ void init_host_matrices_half_NT(half *a, half *b, half *c, int M_GLOBAL, 
    int N_GLOBAL, int K_GLOBAL)
{
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i*K_GLOBAL+j] = __float2half((rand()  % 5)/7.0);
      //a[i*K_GLOBAL+j] = __float2half(1.0f);

      //printf(" %f \n", __half2float(a[i * K_GLOBAL + j]));
      //printf("%f \n", (float)((rand() % 3)));
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i*K_GLOBAL+j] = __float2half((rand()  % 5)/7.0);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] =  __float2half((rand() + t) % 3);
    //c[t] =  __float2half(1.0f);
  }
}



__host__ void matMultiplyOnHost(half *A, half *B, float *C,
    float alpha, float beta,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns)
{
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += __half2float(A[i * numAColumns + k]) * __half2float(B[k *
            numBColumns + j]);
      }

      C[i*numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}
__host__ void matMultiplyOnHost_NT(half *A, half *B, float *C,
    float alpha, float beta,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns)
{
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += __half2float(A[i * numAColumns + k]) * __half2float(B[j *
            numBRows+ k]);
      }

      C[i*numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}
__host__ void matMultiplyOnHost_NT_align(half *A, half *B, float *C,
    float alpha, float beta,
    int numARows, int numAColumns, int lda,
    int numBRows, int numBColumns, int ldb,
    int numCRows, int numCColumns, int ldc)
{
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += __half2float(A[i * lda+ k]) * __half2float(B[j *
            numBRows+ k]);
      }

      C[i*ldc+ j] = temp * alpha + beta * C[i * ldc+ j];
    }
  }
}
__host__ void matMultiplyOnHostHalf_align(half *A, half *B, half *C,
    float alpha, float beta,
    int numARows, int numAColumns,int lda,
    int numBRows, int numBColumns,int ldb,
    int numCRows, int numCColumns, int ldc)
{
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += __half2float(A[i * lda+ k]) * __half2float(B[k * ldb+
            j]);
      }

      C[i*ldc+ j] = __float2half(temp * alpha + beta * __half2float(C[i * ldc + j]));
    }
  }
}

__host__ void matMultiplyOnHostHalf(half *A, half *B, half *C,
    float alpha, float beta,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns)
{
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += __half2float(A[i * numAColumns + k]) * __half2float(B[k * numBColumns+
            j]);
      }

      C[i*numCColumns + j] = __float2half(temp * alpha + beta * __half2float(C[i * numCColumns +
            j]));
    }
  }
}

__host__ void matMultiplyOnHostHalf_NT_align(half *A, half *B, half *C,
    float alpha, float beta,
    int numARows, int numAColumns, int lda,
    int numBRows, int numBColumns, int ldb,
    int numCRows, int numCColumns, int ldc)
{
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += __half2float(A[i * lda+ k]) * __half2float(B[j * ldb +
            k]);
      }

      C[i*ldc+ j] = __float2half((temp * alpha + beta * __half2float(C[i * ldc+
              j])));
    }
  }
}

int test_gemm_wmma_shm_NN(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm ...\n");

  int M, N, K, M_TILES, N_TILES, K_TILES;

  M = 16;  
  N = 16; 
  K = 16;


  M_TILES = M_GLOBAL / M;
  N_TILES = N_GLOBAL / N;
  K_TILES = K_GLOBAL / K;


  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  half *A_h = NULL;
  half *B_h = NULL;
  float *C_h = NULL;

#if CPU_DEBUG
  float *result_hD = NULL;
  float *result_host = NULL;
#endif

  A_h = (half*) malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
  B_h = (half*) malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
  C_h = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

#if CPU_DEBUG
  result_hD   = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  result_host = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

  half *A = NULL;
  half *B = NULL;
  float *C = NULL;
  float *D = NULL;

  checkCudaErrors(cudaMalloc((void**)&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * N_GLOBAL * K_GLOBAL));

  checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices_NN(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

  printf("Preparing data for GPU...\n");

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));


  const int WARPS_PER_BLOCK = 8;
  const int THREADS_PER_BLOCK =  (WARP_SIZE * WARPS_PER_BLOCK);

  const int BLOCK_ROW_WARPS = 2;
  const int BLOCK_COL_WARPS = 4;
  const int WARP_ROW_TILES  = 4;
  const int WARP_COL_TILES  = 2;

  const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS); // 8
  const int BLOCK_COL_TILES  = (WARP_COL_TILES * BLOCK_COL_WARPS); // 8

  const int SKEW_HALF = 8;
  const int CHUNK_K = 4;

  int SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M) * 
      (CHUNK_K * K + SKEW_HALF) * 2,
      M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * 
      (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float));


  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  const float alpha = 1.1f;
  const float beta = 1.2f;

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  // If enough shared memory available on the GPU use high performant kernel
  // if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ)

  printf("Computing... using high performance kernel compute_gemm \n");
  printf("block_num = %d, block_size = %d \n", 
      40, 256);

  printf("SHEM_SZ = %d \n", SHMEM_SZ);  

  checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

  int lda = K_GLOBAL;
  int ldb = N_GLOBAL;
  int ldc = N_GLOBAL;
  int ldd = N_GLOBAL;
#if 1
  for(int run = 0; run < 100; run++){
    checkKernelErrors((_gemm_wmma_shm<<<40, 
          THREADS_PER_BLOCK, SHMEM_SZ>>>(0, 0, M_GLOBAL, N_GLOBAL, K_GLOBAL, 
            alpha, A, lda, B, ldb, beta, C, ldc))); 
  }
#endif



#if CPU_DEBUG
  checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float)*M_GLOBAL*N_GLOBAL, 
        cudaMemcpyDeviceToHost));
#endif

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

  matMultiplyOnHost(A_h, B_h, result_host,
      alpha, beta,
      M_GLOBAL, K_GLOBAL,
      K_GLOBAL, N_GLOBAL,
      M_GLOBAL, N_GLOBAL);

  for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    if (fabs(result_hD[i] - result_host[i]) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], 
          result_host[i]);
  }
  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;

  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds * 0.01/1000.)) / 1e12);

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));
  checkCudaErrors(cudaFree((void*)D));


  return 0;
}

int test_gemm_wmma_shm(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm ...\n");

  int M, N, K, M_TILES, N_TILES, K_TILES;

  M = 16;  
  N = 16; 
  K = 16;


  M_TILES = M_GLOBAL / M;
  N_TILES = N_GLOBAL / N;
  K_TILES = K_GLOBAL / K;


  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

  half *A_h = NULL;
  half *B_h = NULL;
  float *C_h = NULL;

#if CPU_DEBUG
  float *result_hD = NULL;
  float *result_host = NULL;
#endif

  A_h = (half*) malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
  B_h = (half*) malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
  C_h = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

#if CPU_DEBUG
  result_hD   = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  result_host = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

  half *A = NULL;
  half *B = NULL;
  float *C = NULL;
  float *D = NULL;

  checkCudaErrors(cudaMalloc((void**)&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * N_GLOBAL * K_GLOBAL));

  checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

  printf("Preparing data for GPU...\n");

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));


  const int WARPS_PER_BLOCK = 8;
  const int THREADS_PER_BLOCK =  (WARP_SIZE * WARPS_PER_BLOCK);

  const int BLOCK_ROW_WARPS = 2;
  const int BLOCK_COL_WARPS = 4;
  const int WARP_ROW_TILES  = 4;
  const int WARP_COL_TILES  = 2;

  //const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS); // 8
  const int BLOCK_COL_TILES  = (WARP_COL_TILES * BLOCK_COL_WARPS); // 8

  const int SKEW_HALF = 8;
  const int CHUNK_K = 4;

  int SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M) * 
      (CHUNK_K * K + SKEW_HALF) * 2,
      M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * 
      (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float));


  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  const float alpha = 1.1f;
  const float beta = 1.2f;

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  // If enough shared memory available on the GPU use high performant kernel
  // if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ)

  printf("Computing... using high performance kernel compute_gemm \n");
  printf("block_num = %d, block_size = %d \n", 
      40, 256);

  printf("SHEM_SZ = %d \n", SHMEM_SZ);  

  checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

  int lda = K_GLOBAL;
  int ldb = N_GLOBAL;
  int ldc = N_GLOBAL;
  int ldd = N_GLOBAL;
#if 1
  for(int run = 0; run < 100; run++){
    checkKernelErrors((_gemm_wmma_shm<<<40, 
          THREADS_PER_BLOCK, SHMEM_SZ>>>(0, 0, M_GLOBAL, N_GLOBAL, K_GLOBAL, 
            alpha, A, lda, B, ldb, beta, C, ldc)));
  }
#endif



#if CPU_DEBUG
  checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float)*M_GLOBAL*N_GLOBAL, 
        cudaMemcpyDeviceToHost));
#endif

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

  matMultiplyOnHost_NT(A_h, B_h, result_host,
      alpha, beta,
      M_GLOBAL, K_GLOBAL, 
      K_GLOBAL, N_GLOBAL,  
      M_GLOBAL, N_GLOBAL) ;

  for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    if (fabs(result_hD[i] - result_host[i]) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], 
          result_host[i]);
  }
  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;

  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds * 0.01/1000.)) / 1e12);

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));
  checkCudaErrors(cudaFree((void*)D));


  return 0;
}


// default
int test_gemm_wmma_shm_half(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm_half ...\n");

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


  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

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
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * ldb* N_GLOBAL));

  checkCudaErrors(cudaMalloc((void**)&C, sizeof(half) * M_GLOBAL * ldc));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);

  init_host_matrices_half_NT_align(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, 
      lda, ldb, ldc);

  printf("Preparing data for GPU...\n");

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * lda, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * ldb, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(half) * M_GLOBAL * ldc, 
        cudaMemcpyHostToDevice));

  const int WARPS_PER_BLOCK = 8;
  const int THREADS_PER_BLOCK =  (WARP_SIZE * WARPS_PER_BLOCK);

  const int BLOCK_ROW_WARPS = 2;
  const int BLOCK_COL_WARPS = 4;
  const int WARP_ROW_TILES  = 4;
  const int WARP_COL_TILES  = 2;

  const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS); // 8
  const int BLOCK_COL_TILES  = (WARP_COL_TILES * BLOCK_COL_WARPS); // 8

  const int SKEW_HALF = 8;
  const int CHUNK_K = 4;

  int SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M) * 
      (CHUNK_K * K + SKEW_HALF) * 2,
      M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * 
      (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(half));


  printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  const float alpha = 1.0f;
  //const float alpha = 1.1f;
  //const float beta = 1.2f;
  const float beta = 0.0f;

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  // If enough shared memory available on the GPU use high performant kernel
  // if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ)

  printf("Computing... using high performance kernel compute_gemm \n");
  printf("block_num = %d, block_size = %d \n", 
      40, THREADS_PER_BLOCK);

  printf("SHEM_SZ = %d \n", SHMEM_SZ);  

  checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_align, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

  //int lda = K;
  //int ldb = N;
  //int ldc = N;

#if 1
  //for(int run = 0; run < 100; run++)
  {
    checkKernelErrors((_gemm_wmma_shm_half_align<<<40, THREADS_PER_BLOCK, SHMEM_SZ>>>(
            0, 0, M_GLOBAL, N_GLOBAL, K_GLOBAL, 
            alpha, A, lda, B, ldb, beta, C, ldc)));
  }
#endif


#if CPU_DEBUG
  checkCudaErrors(cudaMemcpy(result_hD, C, sizeof(half)*M_GLOBAL*ldc, 
        cudaMemcpyDeviceToHost));
#endif

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds * 0.01/1000.)) / 1e12);


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

#if 0
  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(half) * M_GLOBAL * N_GLOBAL, 
        cudaMemcpyHostToDevice));
  lda = K_GLOBAL;
  ldb = N_GLOBAL;
  ldc = N_GLOBAL;
  gemm_cuBlas(1, 0, A, B, C, alpha, beta, M_GLOBAL, N_GLOBAL,
      K_GLOBAL, lda, ldb, ldc);
  checkCudaErrors(cudaMemcpy(result_host, C, sizeof(half) * M_GLOBAL* N_GLOBAL,
        cudaMemcpyDeviceToHost));
#endif

#if 0
  for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    if (fabs(__half2float(result_hD[i]) - __half2float(result_host[i])) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i,
          __half2float(result_hD[i]), 
          __half2float(result_host[i]));
  }
#endif
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      if (fabs(__half2float(result_hD[i*ldc+j]) -
            __half2float(result_host[i*ldc+j])) > 0.1f)
      {
        printf("mismatch i=%d, j=%d, result_hD=%f result_host=%f\n", i,j,
            __half2float(result_hD[i*ldc+ j]), 
            __half2float(result_host[i*ldc+ j]));
      }
    }
  }

  free(result_hD);
  free(result_host);
#endif

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));


  return 0;
}


int call_gemm_wmma_shm_half(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm_half ...\n");

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


  printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
  printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
  printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

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
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * ldb* N_GLOBAL));

  checkCudaErrors(cudaMalloc((void**)&C, sizeof(half) * M_GLOBAL * ldc));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);

  init_host_matrices_half_NT_align(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, 
      lda, ldb, ldc);

  printf("Preparing data for GPU...\n");

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

  printf("Computing... using high performance kernel compute_gemm \n");
  for(int i = 0; i < 100; i++)
    gemm_wmma_shm_half( 0, 0, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, A, lda, B, ldb, beta, C, ldc);


#if CPU_DEBUG
  checkCudaErrors(cudaMemcpy(result_hD, C, sizeof(half)*M_GLOBAL*ldc, 
        cudaMemcpyDeviceToHost));
#endif

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds  /1e5)) / 1e12);


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

#if 0
  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(half) * M_GLOBAL * N_GLOBAL, 
        cudaMemcpyHostToDevice));
  lda = K_GLOBAL;
  ldb = N_GLOBAL;
  ldc = N_GLOBAL;
  gemm_cuBlas(1, 0, A, B, C, alpha, beta, M_GLOBAL, N_GLOBAL,
      K_GLOBAL, lda, ldb, ldc);
  checkCudaErrors(cudaMemcpy(result_host, C, sizeof(half) * M_GLOBAL* N_GLOBAL,
        cudaMemcpyDeviceToHost));
#endif

#if 0
  for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    if (fabs(__half2float(result_hD[i]) - __half2float(result_host[i])) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i,
          __half2float(result_hD[i]), 
          __half2float(result_host[i]));
  }
#endif
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      if (fabs(__half2float(result_hD[i*ldc+j]) -
            __half2float(result_host[i*ldc+j])) > 0.9f)
      {
        printf("mismatch i=%d, j=%d, result_hD=%f result_host=%f\n", i,j,
            __half2float(result_hD[i*ldc+ j]), 
            __half2float(result_host[i*ldc+ j]));
      }
    }
  }

  free(result_hD);
  free(result_host);
#endif

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));


  return 0;
}

int call_gemm_wmma_shm_half_config(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm_half_config ...\n");

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

  printf("Preparing data for GPU...\n");

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

  printf("Computing... gemm_wmma_shm_half_config \n");
  int runtimes = 1;
  for(int i = 0; i < runtimes; i++)
    gemm_wmma_shm_half_config(false, false, M_GLOBAL, N_GLOBAL, K_GLOBAL, 
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
      }
    }
  }

  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));


  return 0;
}

//
int call_gemm_wmma_shm_half_128_16(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm_half_128_16 ...\n");

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

  printf("Preparing data for GPU...\n");

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

  printf("Computing... gemm_wmma_shm_half_128_16 \n");
  int runtimes = 100;
  //for(int i = 0; i < runtimes; i++)
  gemm_wmma_shm_half_128_16(false, false, 
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

//
int call_gemm_wmma_shm_r_opt(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm_r_opt ...\n");

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

  printf("Preparing data for GPU...\n");

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

  printf("Computing...........  \n");
  int runtimes = 100;
  //for(int i = 0; i < runtimes; i++)
  gemm_wmma_shm_r_opt(false, false, 
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
#if 1
      if (fabs(__half2float(result_hD[i*ldc+j]) -
            __half2float(result_host[i*ldc+j])) > 0.9f)
#endif
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


//
int call_gemm_wmma_shm_splitk_persistent(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm_splitk_persistent ...\n");

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

  printf("Computing... gemm_wmma_shm_split_persistent \n");
  int runtimes = 100;
  //for(int i = 0; i < runtimes; i++)
  gemm_wmma_shm_splitk_persistent(false, false, 
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




int call_gemm_wmma_shm_persistent_r(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("call_gemm_wmma_shm_persistent_r ...\n");

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

  printf("Computing.......   \n");
  int runtimes = 100;
  //for(int i = 0; i < runtimes; i++)
  gemm_wmma_shm_persistent_r(false, false, 
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
#if 1
      if (fabs(__half2float(result_hD[i*ldc+j]) -
            __half2float(result_host[i*ldc+j])) > 0.9f)
#endif
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


//
int call_gemm_wmma_shm_persistent_db(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_shm_persistent_db ...\n");

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
  gemm_wmma_shm_persistent_db(false, false, 
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


// 
int test_gemm_wmma(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma ...\n");

  const int WMMA_M = 16;  
  const int WMMA_N = 16; 
  const int WMMA_K = 16; 

  printf("M: %d \n", M_GLOBAL);
  printf("N: %d \n", N_GLOBAL);
  printf("K: %d \n", K_GLOBAL);

  half *A_h = NULL;
  half *B_h = NULL;
  float *C_h = NULL;

#if CPU_DEBUG
  float *result_hD = NULL;
  float *result_host = NULL;
#endif

  A_h = (half*)  malloc(sizeof(half)  * M_GLOBAL * K_GLOBAL);
  B_h = (half*)  malloc(sizeof(half)  * K_GLOBAL * N_GLOBAL);
  C_h = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

#if CPU_DEBUG
  result_hD   = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  result_host = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

  half *A = NULL;
  half *B = NULL;
  float *C = NULL;
  float *D = NULL;

  checkCudaErrors(cudaMalloc((void**)&A, sizeof(half)  * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half)  * N_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

  printf("Preparing data for GPU...\n");

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

  const float alpha = 1.1f;
  const float beta = 1.2f;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  printf("Computing... using gemm_wmma kernel\n");
  printf("grid_x = %d, grid_y = %d, blk_x = %d, blk_y = %d \n", gridDim.x, 
      gridDim.y, blockDim.x, blockDim.y);

  int runtimes = 100;
  for(int run = 0; run < runtimes; run++)
  {
#if 1
    checkKernelErrors((_gemm_wmma<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, 
            N_GLOBAL, K_GLOBAL, alpha, beta)));
#endif
  }

#if CPU_DEBUG
  checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, 
        cudaMemcpyDeviceToHost));
#endif

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

  matMultiplyOnHost(A_h, B_h, result_host,
      alpha, beta,
      M_GLOBAL, K_GLOBAL,
      K_GLOBAL, N_GLOBAL,
      M_GLOBAL, N_GLOBAL);

  for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
    if (fabs(result_hD[i] - result_host[i]) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], 
          result_host[i]);
  }
  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;

  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds / 1000.)) / 1e12* runtimes);

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));
  checkCudaErrors(cudaFree((void*)D));


  return 0;
}

int test_gemm_wmma_half(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL)
{
  printf("test_gemm_wmma_half ...\n");

  const int WMMA_M = 16;  
  const int WMMA_N = 16; 
  const int WMMA_K = 16; 

  printf("M: %d \n", M_GLOBAL);
  printf("N: %d \n", N_GLOBAL);
  printf("K: %d \n", K_GLOBAL);

  half *A_h = NULL;
  half *B_h = NULL;
  half *C_h = NULL;

#if CPU_DEBUG
  half *result_hD = NULL;
  half *result_host = NULL;
#endif

  A_h = (half*)  malloc(sizeof(half)  * M_GLOBAL * K_GLOBAL);
  B_h = (half*)  malloc(sizeof(half)  * K_GLOBAL * N_GLOBAL);
  C_h = (half*)  malloc(sizeof(half)  * M_GLOBAL * N_GLOBAL);

#if CPU_DEBUG
  result_hD   = (half*) malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
  result_host = (half*) malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
#endif

  half *A = NULL;
  half *B = NULL;
  half *C = NULL;
  half *D = NULL;

  checkCudaErrors(cudaMalloc((void**)&A, sizeof(half)  * M_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half)  * N_GLOBAL * K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&C, sizeof(half) * M_GLOBAL * N_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&D, sizeof(half) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices_half(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

  printf("Preparing data for GPU...\n");

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(half) * M_GLOBAL * N_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(D, 0, sizeof(half) * M_GLOBAL * N_GLOBAL));

  const float alpha = 1.1f;
  const float beta = 0.0f;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  printf("Computing... using gemm_wmma kernel\n");
  printf("grid_x = %d, grid_y = %d, blk_x = %d, blk_y = %d \n", gridDim.x, 
      gridDim.y, blockDim.x, blockDim.y);

  for(int run = 0; run < 100; run++)
  {
#if 1
    checkKernelErrors((_gemm_wmma_half<<<gridDim, blockDim>>>(0, 0, A, B, C, D, 
            M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta, K_GLOBAL, N_GLOBAL,
            N_GLOBAL)));
#endif
  }

#if CPU_DEBUG
  //checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(half) * M_GLOBAL * N_GLOBAL, 
  //      cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(result_hD, C, sizeof(half) * M_GLOBAL * N_GLOBAL, 
        cudaMemcpyDeviceToHost));
#endif

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(half) * M_GLOBAL * N_GLOBAL);

  matMultiplyOnHostHalf(A_h, B_h, result_host,
      alpha, beta,
      M_GLOBAL, K_GLOBAL,
      K_GLOBAL, N_GLOBAL,
      M_GLOBAL, N_GLOBAL);

  for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
    if (fabs(__half2float(result_hD[i]) - __half2float(result_host[i])) > 0.1f)
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i,
          __half2float(result_hD[i]), 
          __half2float(result_host[i]));
  }
  free(result_hD);
  free(result_host);
#endif

  float milliseconds = 0;

  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds * 0.01/1000.)) / 1e12);

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));
  checkCudaErrors(cudaFree((void*)D));


  return 0;
}

int call_gemm_wmma_half(int M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int lda, int
    ldb, int ldc)
{
  printf("call_gemm_wmma_half ...\n");

  half *A_h = NULL;
  half *B_h = NULL;
  half *C_h = NULL;

#if CPU_DEBUG
  half *result_hD = NULL;
  half *result_host = NULL;
#endif

  A_h = (half*)  malloc(sizeof(half)  * M_GLOBAL * lda);
  B_h = (half*)  malloc(sizeof(half)  * K_GLOBAL * ldb);
  C_h = (half*)  malloc(sizeof(half)  * M_GLOBAL * ldc);

#if CPU_DEBUG
  result_hD   = (half*) malloc(sizeof(half) * M_GLOBAL * ldc);
  result_host = (half*) malloc(sizeof(half) * M_GLOBAL * ldc);
#endif

  half *A = NULL;
  half *B = NULL;
  half *C = NULL;
  half *D = NULL;

  checkCudaErrors(cudaMalloc((void**)&A, sizeof(half)  * M_GLOBAL * lda));
  checkCudaErrors(cudaMalloc((void**)&B, sizeof(half)  * ldb* K_GLOBAL));
  checkCudaErrors(cudaMalloc((void**)&C, sizeof(half) * M_GLOBAL * ldc));
  checkCudaErrors(cudaMalloc((void**)&D, sizeof(half) * M_GLOBAL * ldc));

  assert(((unsigned long long)A) % 128 == 0);
  assert(((unsigned long long)B) % 128 == 0);
  assert(((unsigned long long)C) % 128 == 0);
  assert(((unsigned long long)D) % 128 == 0);

  init_host_matrices_half_align(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL, lda, ldb, ldc);

#if 0
  char*   fileNameB = "1/weights_c_h_480_0.check";
  char*   fileNameA = "1/xt_h_2_480_0.check";
  init_host_matrices_half_align_file(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL,
      K_GLOBAL, lda, ldb, ldc, fileNameA,  fileNameB);

  printf("Preparing data for GPU...\n");
#endif

  checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * lda, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * ldb* K_GLOBAL, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(C, C_h, sizeof(half) * M_GLOBAL * ldc, 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemset(D, 0, sizeof(half) * M_GLOBAL * ldc));

  const float alpha = 1.1f;
  const float beta = 0.0f;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));
  int runtimes = 100;

  for(int i =0 ;i < runtimes;  i++){
    gemm_wmma_half(0, 0, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, A, lda, 
        B, ldb, beta, C, ldc);
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("gemm_wmma Time: %f ms\n", milliseconds);

#if CPU_DEBUG
  checkCudaErrors(cudaMemcpy(result_hD, C, sizeof(half) * M_GLOBAL * ldc, 
        cudaMemcpyDeviceToHost));
#endif


#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");

  memcpy(result_host, C_h, sizeof(half) * M_GLOBAL * ldc);
#if 0
  matMultiplyOnHostHalf_align(A_h, B_h, result_host,
      alpha, beta,
      M_GLOBAL, K_GLOBAL, lda,
      K_GLOBAL, N_GLOBAL, ldb,
      M_GLOBAL, N_GLOBAL, ldc);
#endif
#if 1
  cudaMemcpy(C, C_h, sizeof(half)*M_GLOBAL*ldc, cudaMemcpyHostToDevice);
  matMultiplyOnCublas(B, A, C, alpha, beta, N_GLOBAL, M_GLOBAL,
      K_GLOBAL,ldb,lda,ldc);
  //gemm_cuBlas(1, 0, B, A, C, alpha, beta, N_GLOBAL, M_GLOBAL,
  //    K_GLOBAL,ldb,lda,ldc);
  cudaMemcpy(result_host, C, sizeof(half)*M_GLOBAL*ldc, cudaMemcpyDeviceToHost);
#endif
#if 1 
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      if (fabs(__half2float(result_hD[i*ldc+j]) -
            __half2float(result_host[i*ldc+j])) > 0.2f)
      {
        printf("mismatch i=%d, j=%d, result_hD=%f result_host=%f\n", i,j,
            __half2float(result_hD[i*ldc+ j]), 
            __half2float(result_host[i*ldc+ j]));
      }
    }
  }
#endif
  free(result_hD);
  free(result_host);
#endif

  printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL *
          2)/(milliseconds/1000.)) / 1e12 * runtimes);

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree((void*)A));
  checkCudaErrors(cudaFree((void*)B));
  checkCudaErrors(cudaFree((void*)C));
  checkCudaErrors(cudaFree((void*)D));

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
  M = 128; N = 512; K = 512;
  lda = K+32, ldb = N+64, ldc = N+128;
  M = 16;
  for(int i =0 ; i < 1; i++){
    //call_gemm_wmma_shm_half(M, N, K); // ok

    //M = 32; N =6144; K = 16;
    M = 128; N = 3072; K = 1024;
    //M = 128; N = 512; K = 512;
    //M = 128; N = 256; K = 512;
    //call_gemm_wmma_shm_half(M, N, K); // best for gemm_r

    //call_gemm_wmma_shm_half_config(M, N, K);
    //call_gemm_wmma_shm_half_128_16(M, N, K);
    //call_gemm_wmma_shm_r_opt(M, N, K);

    //call_gemm_wmma_shm_persistent(M, N, K);
    //call_gemm_wmma_shm_splitk_persistent(M, N, K);
    //call_gemm_wmma_shm_persistent_db(M, N, K);
    //M = 128; N = 3072; K = 1024;
    call_gemm_wmma_shm_persistent_r(M, N, K);

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
