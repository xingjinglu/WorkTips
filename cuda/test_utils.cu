#include <cstdio>
#include <assert.h>
#include <fstream>
#include <random>

#include <cuda.h>
#include <mma.h>
#include <cublas_v2.h>

#include "gemm_utils.h"
#include "gemm_wmma.cuh"

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

