#ifndef _TEST_UTILS_CUH_
#define _TEST_UTILS_CUH_

__host__ void init_host_matrices_NN(half *a, half *b, float *c, int M_GLOBAL, 
    int N_GLOBAL, int K_GLOBAL);

__host__ void init_host_matrices(half *a, half *b, float *c, int M_GLOBAL, 
    int N_GLOBAL, int K_GLOBAL);

__host__ void init_host_matrices_half_align_file(half *a, half *b, half *c, int
    M_GLOBAL, int N_GLOBAL, int K_GLOBAL,int lda, int ldb, int ldc, char*
    fileNameA , char* fileNameB);

__host__ void init_host_matrices_half_align(half *a, half *b, half *c, int
M_GLOBAL, int N_GLOBAL, int K_GLOBAL,int lda, int ldb, int ldc);


__host__ void init_host_matrices_half(half *a, half *b, half *c, int M_GLOBAL, 
    int N_GLOBAL, int K_GLOBAL);

__host__ void init_host_matrices_half_NT_align(half *a, half *b, half *c, int
M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int lda, int ldb, int ldc);


__host__ void init_host_matrices_half_NT(half *a, half *b, half *c, int
M_GLOBAL, int N_GLOBAL, int K_GLOBAL);



__host__ void matMultiplyOnHost(half *A, half *B, float *C,
    float alpha, float beta,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns);


__host__ void matMultiplyOnHost_NT(half *A, half *B, float *C,
    float alpha, float beta,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns);

__host__ void matMultiplyOnHost_NT_align(half *A, half *B, float *C,
    float alpha, float beta,
    int numARows, int numAColumns, int lda,
    int numBRows, int numBColumns, int ldb,
    int numCRows, int numCColumns, int ldc);

__host__ void matMultiplyOnHostHalf_align(half *A, half *B, half *C,
    float alpha, float beta,
    int numARows, int numAColumns,int lda,
    int numBRows, int numBColumns,int ldb,
    int numCRows, int numCColumns, int ldc);


__host__ void matMultiplyOnHostHalf(half *A, half *B, half *C,
    float alpha, float beta,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns);


__host__ void matMultiplyOnHostHalf_NT_align(half *A, half *B, half *C,
    float alpha, float beta,
    int numARows, int numAColumns, int lda,
    int numBRows, int numBColumns, int ldb,
    int numCRows, int numCColumns, int ldc);


#endif
