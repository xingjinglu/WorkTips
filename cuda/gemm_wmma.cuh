#ifndef _PERSISTENTWAVERNN_KERNEL_H
#define _PERSISTENTWAVERNN_KERNEL_H

#include "configure.h"

#define WARP_SIZE 32
#define warpWize 32
#define warp_size 32

#ifndef MAX                                                                     
#define MAX(a, b) ((a) > (b) ? (a) : (b))                                               
#endif  

//
__global__ void _gemm_wmma_shm(bool transa, bool transb,
    int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    float *C, int ldc);
__global__ void _gemm_wmma_shm_half(bool transa, bool transb,
    int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);

template<const int wmma_m, const int wmma_n, const int wmma_k, const int
warp_row_tiles, const int warp_col_tiles>
__global__ void _gemm_wmma_shm_half_config(
    int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, 
    int WARPS_PER_BLOCK, //const int warp_size,
    int CHUNK_K,
    //bool transa, bool transb,
     int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);

template<const int wmma_m, const int wmma_n, const int wmma_k, const int
warp_row_tiles, const int warp_col_tiles>
__global__ void _gemm_wmma_shm_half_128_16(
    int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, 
    int WARPS_PER_BLOCK, //const int warp_size,
    int CHUNK_K,
    //bool transa, bool transb,
     int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);

// 
template<const int wmma_m, const int wmma_n, const int wmma_k, const int
warp_row_tiles, const int warp_col_tiles>
__global__ void _gemm_wmma_shm_half_128_16(
    int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, 
    int WARPS_PER_BLOCK, //const int warp_size,
    int CHUNK_K,
    //bool transa, bool transb,
     int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);

template<const int WmmaM, const int WmmaN, const int WmmaK, const int
WarpRowTiles, const int WarpColTiles, const int ChunkCol, int ChunkK>
__global__ void _gemm_wmma_shm_persistent(
    int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, 
    int WARPS_PER_BLOCK, //const int warp_size,
    //bool transa, bool transb,
     int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);

template<const int WmmaM, const int WmmaN, const int WmmaK, const int
WarpRowTiles, const int WarpColTiles, const int ChunkCol, int ChunkK>
__global__ void _gemm_wmma_shm_splitk_persistent(
    int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, 
    int WARPS_PER_BLOCK, //const int warp_size,
    //bool transa, bool transb,
     int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);



template<const int WmmaM, const int WmmaN, const int WmmaK, const int
WarpRowTiles, const int WarpColTiles, const int ChunkCol, int ChunkK>
__global__ void _gemm_wmma_shm_persistent_r(
    int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, 
    int WARPS_PER_BLOCK, //const int warp_size,
    //bool transa, bool transb,
     int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);


template<const int WmmaM, const int WmmaN, const int WmmaK, const int
WarpRowTiles, const int WarpColTiles, const int ChunkCol, int ChunkK>
__global__ void _gemm_wmma_shm_persistent_db(
    int BLOCK_ROW_WARPS, int BLOCK_COL_WARPS, 
    int WARPS_PER_BLOCK, //const int warp_size,
    //bool transa, bool transb,
     int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);


// Deprecated function.
__global__ void _gemm_wmma_shm_half_align(bool transa, bool transb,
    int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    half *C, int ldc);

__global__ void _gemm_wmma_shm_NN(int M, int N, int K, const half *A,
    const half *B, const float *C, float *D, float alpha, float beta);


// 
__global__ void _gemm_wmma(half *a, half *b, float *c, float *d, int m_ld, 
    int n_ld, int k_ld, float alpha, float beta);

__global__ void _gemm_wmma_half(bool OP_A_T, bool OP_B_T, half *a, half *b, 
    half *c, half *d, int m_ld, 
    int n_ld, int k_ld, float alpha, float beta,
    int lda, int ldb, int ldc);

int gemm_wmma_half(bool OP_A_N, bool OP_B_N, int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);


// 
int gemm_wmma_shm_half(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);

int gemm_wmma_shm_half_config(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);

int gemm_wmma_shm_half_128_16(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);

int gemm_wmma_shm_r_opt(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);

int gemm_wmma_shm_persistent(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);

int gemm_wmma_shm_splitk_persistent(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);

int gemm_wmma_shm_persistent_r(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);

int gemm_wmma_shm_persistent_db(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc);



#endif
