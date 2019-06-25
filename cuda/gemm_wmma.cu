#include <cstdio>
#include <cassert>

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <mma.h>
#include <cuda.h>
#include <cublas_v2.h>

#include "gemm_wmma.cuh"
#include "gemm_utils.h"
#include "configure.h"

//#include "WaveRNNKernel.h"
//#include "cu-kernel.h"
/*****************
// the gemm_wmma_shm the M_GLOBAL should be timeof 16
// the N_GLOBAL should be times of 64
// the K_GLOBAL should be times of 64
 ********************/

// gemm_tc configures.
#ifndef SHARED_MEMORY_LIMIT_64K
#define SHARED_MEMORY_LIMIT_64K 1
#endif

#define C_LAYOUT wmma::mem_row_major


using namespace nvcuda;


// default: NT
__global__ void _gemm_wmma_shm(bool transa, bool transb,
    int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb,
    float beta,
    float *C, int ldc)
{
  // kernel configures.
  const int CHUNK_K = 4;
  const int SKEW_HALF = 8;
  const int WARPS_PER_BLOCK = 8;
  const int BLOCK_ROW_WARPS = 2;
  const int BLOCK_COL_WARPS = 4;
  const int WARP_ROW_TILES = 4;
  const int WARP_COL_TILES = 2;

  const int M = 16;
  const int N = 16;
  const int K = 16;

  const int  M_TILES = M_GLOBAL / M;
  const int  N_TILES = N_GLOBAL / N;
  const int  K_TILES = K_GLOBAL / K;

  //const int  THREADS_PER_BLOCK =  WARP_SIZE * WARPS_PER_BLOCK; // 256
  const int  CHUNK_LINE_BYTES =  CHUNK_K * K * sizeof(half);   // 128
  const int  WARP_COPY_BYTES = WARP_SIZE * sizeof(int4);       // 512
  // 4
  const int  CHUNK_COPY_LINES_PER_WARP = WARP_COPY_BYTES / CHUNK_LINE_BYTES; 
  // 8
  const int  CHUNK_COPY_LINE_LANES = WARP_SIZE / CHUNK_COPY_LINES_PER_WARP;

  const int  BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS; // 8
  const int  BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS; // 8

  const int  GLOBAL_MEM_STRIDE = N_GLOBAL;

  const int  SHMEM_STRIDE = N * BLOCK_ROW_TILES;
  const int  SHMEM_OFFSET = N * WARP_ROW_TILES;


  // 
  extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp 
  // computes.
  float *shmem_warp_tile_ptr = (float*)&shmem[0][0] + (warpId/2) * 
    SHMEM_STRIDE * K * 2 + (warpId%2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile 
  // to and from shared memory.
  float *shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId * 
    SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may result
  // in a loss of precision). Zero still needs to be specially handled though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
  // right and down, and selects the next tile to compute. Once there's no such tile,
  // all warps in this CTA exit.
  for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared memory.
    const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + 
      block_tile_j * N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < K; i++) {
      typedef int4 copy_t;

      *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = 
        *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment multiplications
    // along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2) :
      (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
      size_t shmem_idx = warpId < (WARPS_PER_BLOCK/2) ? (M * (warpId % (WARPS_PER_BLOCK/2)) * 2) : 
        (N * (warpId % (WARPS_PER_BLOCK/2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4*)(warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) + (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for(int i = 0; i < ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
        // Copy 16 bytes at once in each lane.
        *((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int4*)((half*)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId/2) * M * 2 + (i * M);
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be reused
              // against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
        // warp are well-defined even though element indices within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++)
          c[i][j].x[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global 
    // memory.
    float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((int4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
        *((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}


//  Deprecated function.
//  _gemm_wmma_shm_half support align in default.
__global__ void _gemm_wmma_shm_half_align(bool transa, bool transb, 
    int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{

  // kernel configures.
  const int CHUNK_K = 4;
  const int SKEW_HALF = 8;

  const int WARPS_PER_BLOCK = 8;

  const int BLOCK_ROW_WARPS = 2;
  const int BLOCK_COL_WARPS = 4;

  const int WARP_ROW_TILES = 4;
  const int WARP_COL_TILES = 2;

  const int M = 16;
  const int N = 16;
  const int K = 16;

  const int  M_TILES = M_GLOBAL / M;
  const int  N_TILES = N_GLOBAL / N;
  const int  K_TILES = K_GLOBAL / K;

  //const int  THREADS_PER_BLOCK =  WARP_SIZE * WARPS_PER_BLOCK;

  const int  CHUNK_LINE_BYTES =  CHUNK_K * K * sizeof(half);  // 128
  const int  WARP_COPY_BYTES = WARP_SIZE * sizeof(int4);      // 512
  // 4 
  const int  CHUNK_COPY_LINES_PER_WARP = WARP_COPY_BYTES / CHUNK_LINE_BYTES;
  // 8 Lanes copy ONE_CHUNK_LINE
  const int  CHUNK_COPY_LINE_LANES = WARP_SIZE / CHUNK_COPY_LINES_PER_WARP;

  const int  BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS; // 8
  const int  BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS; // 8

  //const int  GLOBAL_MEM_STRIDE = N_GLOBAL;
  const int  GLOBAL_MEM_STRIDE =  ldc;

  const int  SHMEM_STRIDE = N * BLOCK_ROW_TILES;  // 128
  const int  SHMEM_OFFSET = N * WARP_ROW_TILES;   // 64


  // 
  extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp 
  // computes.
  half *shmem_warp_tile_ptr = (half*)&shmem[0][0] + (warpId/2) * 
    SHMEM_STRIDE * K * 2 + (warpId%2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile 
  // to and from shared memory.
  half *shmem_warp_stream_ptr = (half*)&shmem[0][0] + warpId * 
    SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may result
  // in a loss of precision). Zero still needs to be specially handled though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) 
      * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared 
    // memory.
    const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + 
      block_tile_j * N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < K; i++) {
      typedef short4 copy_t;
      //if( laneId < 16)
      {
        *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = 
          *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
      }
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment multiplications
    // along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, float> acc[WARP_COL_TILES][WARP_ROW_TILES];
    wmma::fragment<wmma::accumulator, M, N, K, half> c[WARP_COL_TILES][WARP_ROW_TILES];
#if 0
    for(int i = 0;  i < 2; i++)
      for(int j = 0; j < 4; j++)
        wmma::fill_fragment(acc[i][j], 0.0f);
#endif

    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
#if 0
    const half *warp_ptr = (warpId < 4) ? 
      (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2) :
      (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % 4) * 2);
#endif

    const half *warp_ptr = (warpId < 4) ? 
      (&A[block_tile_i * M * lda] + M * lda * (warpId % 4) * 2) :
      (&B[block_tile_j * N * ldb] + N * ldb * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest 
      // copy the B matrix.
      size_t shmem_idx = warpId < (WARPS_PER_BLOCK/2) ? 
        (M * (warpId % (WARPS_PER_BLOCK/2)) * 2) : 
        (N * (warpId % (WARPS_PER_BLOCK/2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (warpId < 4) ? (int4*)(warp_ptr + tile_k * K + 
          //(laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) + 
        (laneId / CHUNK_COPY_LINE_LANES) * lda) + 
        (laneId % CHUNK_COPY_LINE_LANES) :
        (int4*)(warp_ptr + tile_k * K + 
            //(laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) + 
        (laneId / CHUNK_COPY_LINE_LANES) * ldb) + 
        (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for(int i = 0; i < ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
        // Copy 16 bytes at once in each lane.
        *((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        //lane_ptr = (int4*)((half*)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        if(warpId < 4)
          lane_ptr = (int4*)((half*)lane_ptr + lda * CHUNK_COPY_LINES_PER_WARP);
        else
          lane_ptr = (int4*)((half*)lane_ptr + ldb * CHUNK_COPY_LINES_PER_WARP);

        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId/2) * M * 2 + (i * M);
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be reused
              // against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
            }

            wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
          }
        }
      }

      __syncthreads();
    }

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
        // warp are well-defined even though element indices within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++)
          c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);

        half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global memory.
    //float *dst_gmem_warp_stream_ptr = &D[gmem_idx];
    half *dst_gmem_warp_stream_ptr = &C[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((short4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
        *((short4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}

//  Task partition: block(c[128, 128]), warp[c[16, 16])
//  Thread layout: block(256), warp(32) 
//  Mapping between Task
//  |<------------  warp0 ----------------->|<---- warp1 ---------->|
//  task[0,0] task[0,1] task[0,2] task[0,3] | task[0,4] ... task[0,7]
//  task[1,0] task[1,1] task[0,2] task[0,3] | task[0,4] ... task[1,7]
//  -----------------------------------------------------------------
//    .                                     |
//    .
//    .
//  -----------------------------------------------------------------
//  task[6,0] task[6,1]  ... task[6,7]      |
//  task[7,0] task[7,1]  ... task[7,7]      |
//  |<------------  warp6 ----------------->|<---- warp7 ---------->|
//  (transa, transb): (0,0) -> NN
__global__ void _gemm_wmma_shm_half(bool transa, bool transb, 
    int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{

  // kernel configures.
  const int CHUNK_K = 4;
  const int SKEW_HALF = 8;

  const int WARPS_PER_BLOCK = 8;

  const int BLOCK_ROW_WARPS = 2;
  const int BLOCK_COL_WARPS = 4;

  const int WARP_ROW_TILES = 4;
  const int WARP_COL_TILES = 2;

  const int M = 16;
  const int N = 16;
  const int K = 16;

  const int  M_TILES = M_GLOBAL / M;
  const int  N_TILES = N_GLOBAL / N;
  const int  K_TILES = K_GLOBAL / K;

  //const int  THREADS_PER_BLOCK =  WARP_SIZE * WARPS_PER_BLOCK;

  const int  CHUNK_LINE_BYTES =  CHUNK_K * K * sizeof(half);  // 128
  const int  WARP_COPY_BYTES = WARP_SIZE * sizeof(int4);      // 512
  // 4 
  const int  CHUNK_COPY_LINES_PER_WARP = WARP_COPY_BYTES / CHUNK_LINE_BYTES;
  // 8 Lanes copy ONE_CHUNK_LINE
  const int  CHUNK_COPY_LINE_LANES = WARP_SIZE / CHUNK_COPY_LINES_PER_WARP;

  const int  BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS; // 8
  const int  BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS; // 8

  //const int  GLOBAL_MEM_STRIDE = N_GLOBAL;
  const int  GLOBAL_MEM_STRIDE =  ldc;

  const int  SHMEM_STRIDE = N * BLOCK_ROW_TILES;  // 128
  const int  SHMEM_OFFSET = N * WARP_ROW_TILES;   // 64


  // 
  extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp 
  // computes.
  half *shmem_warp_tile_ptr = (half*)&shmem[0][0] + (warpId/2) * 
    SHMEM_STRIDE * K * 2 + (warpId%2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile 
  // to and from shared memory.
  half *shmem_warp_stream_ptr = (half*)&shmem[0][0] + warpId * 
    SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may result
  // in a loss of precision). Zero still needs to be specially handled though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) 
      * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared 
    // memory.
    const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + 
      block_tile_j * N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];


    wmma::fragment<wmma::accumulator, M, N, K, float> acc[WARP_COL_TILES][WARP_ROW_TILES];
    wmma::fragment<wmma::accumulator, M, N, K, half> c[WARP_COL_TILES][WARP_ROW_TILES];
    if(beta != 0.0f){
      // Stream multiple C tiles to shared memory.
#pragma unroll
      for (int i = 0; i < K; i++) {
        typedef short4 copy_t;
        //if( laneId < 16)
        {
          *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = 
            *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
        }
      }

      __syncthreads();

      // These fragments will accumulate the result of A and B matrix fragment multiplications
      // along the K_GLOBAL dimension.
      //wmma::fragment<wmma::accumulator, M, N, K, float> acc[WARP_COL_TILES][WARP_ROW_TILES];
      //wmma::fragment<wmma::accumulator, M, N, K, half> c[WARP_COL_TILES][WARP_ROW_TILES];
#if 0
      for(int i = 0;  i < 2; i++)
        for(int j = 0; j < 4; j++)
          wmma::fill_fragment(acc[i][j], 0.0f);
#endif

      // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
      for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
        for (int j = 0; j < WARP_ROW_TILES; j++) {
          const half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

          wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
        }
      }

      __syncthreads();

      // Scale the C matrix.
#pragma unroll
      for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
        for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
          for (int t = 0; t < c[i][j].num_elements; t++) {
            acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
          }
        }
      }
    }else{
      for(int i = 0;  i < WARP_COL_TILES; i++)
        for(int j = 0; j < WARP_ROW_TILES; j++)
          wmma::fill_fragment(acc[i][j], 0.0f);
    }


    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr = (warpId < 4) ? 
      (&A[block_tile_i * M * lda] + M * lda * (warpId % 4) * 2) :
      (&B[block_tile_j * N * ldb] + N * ldb * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {

      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest 
      // copy the B matrix.
      size_t shmem_idx = warpId < (WARPS_PER_BLOCK/2) ? 
        (M * (warpId % (WARPS_PER_BLOCK/2)) * 2) : 
        (N * (warpId % (WARPS_PER_BLOCK/2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (warpId < 4 ) ? (int4*)(warp_ptr + tile_k * K + 
          (laneId / CHUNK_COPY_LINE_LANES) * lda) + 
        (laneId % CHUNK_COPY_LINE_LANES) :
        (int4*)(warp_ptr + tile_k * K + 
            (laneId / CHUNK_COPY_LINE_LANES) * ldb) + 
        (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for(int i = 0; i < ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
        // Copy 16 bytes at once in each lane.
        *((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        //lane_ptr = (int4*)((half*)lane_ptr + K_GLOBAL* CHUNK_COPY_LINES_PER_WARP);
        if(warpId < 4)
          lane_ptr = (int4*)((half*)lane_ptr + lda * CHUNK_COPY_LINES_PER_WARP);
        else
          lane_ptr = (int4*)((half*)lane_ptr + ldb * CHUNK_COPY_LINES_PER_WARP);

        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId/2) * M * 2 + (i * M);
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be reused
              // against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
            }

            wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
          }
        }
      }

      __syncthreads();
    }

    // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
        // warp are well-defined even though element indices within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++)
          c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);

        half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global memory.
    //float *dst_gmem_warp_stream_ptr = &D[gmem_idx];
    half *dst_gmem_warp_stream_ptr = &C[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((short4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
        *((short4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}






//#define ORIG_C 
//#define ORIG_AB 
template<int WMMA_M,  int WMMA_N, int WMMA_K, int 
WARP_COL_TILES, int WARP_ROW_TILES>
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
    half *C, int ldc)
{

  int SKEW_HALF = 8;
  //const int WARPS_PER_BLOCK = 8;
  //int warp_size = 32;
  int  M_TILES = M_GLOBAL / WMMA_M;
  int  N_TILES = N_GLOBAL / WMMA_N;
  int  K_TILES = K_GLOBAL / WMMA_K;


  int BLOCK_ROW_TILES = BLOCK_ROW_WARPS * WARP_ROW_TILES; 
  int BLOCK_COL_TILES = BLOCK_COL_WARPS * WARP_COL_TILES; 

  // Info of CHUNK_K- A/B.
  int CHUNK_LINE_BYTES =  CHUNK_K * WMMA_K * sizeof(half);      
  int WARP_COPY_BYTES = warp_size * 16; //sizeof(int4);           
  int CHUNK_COPY_LINES_PER_WARP = WARP_COPY_BYTES / CHUNK_LINE_BYTES;  
  int CHUNK_COPY_LINE_LANES = CHUNK_LINE_BYTES / 16; //sizeof(int4);   


  //int GLOBAL_MEM_STRIDE =  ldc;
  int SHMEM_STRIDE = WMMA_N * BLOCK_ROW_TILES;   // C_LINE_LEN of block.
  int SHMEM_OFFSET = WMMA_N * WARP_ROW_TILES;    // C_LINE_LEN of warp.


  // Used for A/B.
  //extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];
  int SHMEM_CHUNK_LINE = CHUNK_K * WMMA_K + SKEW_HALF;
  extern __shared__ half shm[];

  // Warp and lane identification.
  //const int tId = threadIdx.x;
  int tId = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpId = tId / WARP_SIZE;
  //unsigned int laneId = tId % WARP_SIZE;
  unsigned int laneId = tId & WARP_SIZE - 1;
  unsigned int warpX = warpId / BLOCK_ROW_WARPS;
  //unsigned int warpY = warpId % BLOCK_ROW_WARPS;
  unsigned int warpY = warpId & BLOCK_ROW_WARPS - 1;

  // Offset in shared memory from which the B matrix is stored.
  size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;


  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may 
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int grid1D = gridDim.x * gridDim.y;

  //for(unsigned int block_pos = blockIdx.x; ; block_pos += gridDim.x) {
  for(unsigned int block_pos = blockId; ; block_pos += grid1D) {

    unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) 
      * (BLOCK_COL_TILES);
    unsigned int block_tile_j = (block_pos * BLOCK_ROW_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // c-compute-task partion among warps.
    // warps_layout(x, y) = (2, 4), (4, 2), (1, 8)
    // warp0 warp1 warp2 warp3    warp0 warp1   warp0 warp1 ... warp7
    // warp4 warp5 warp6 warp7    warp2 warp3
    //                            warp4 warp5
    //                            warp6 warp7

    // This pointer is used to access the C and D matrix tiles this warp 
    //half *shmem_warp_tile_ptr = (half*)&shmem[0][0] + (warpId / 2) * 
    //  SHMEM_STRIDE * WMMA_N * 2 + (warpId%2) * SHMEM_OFFSET;
    half *shmem_warp_tile_ptr = (half*)&shm[0]+ warpX * 
      WARP_COL_TILES * WMMA_N * SHMEM_STRIDE  + warpY * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile 
    // to and from shared memory. Read from row to cols, warp_read_tile: 
    // blk_row_tiles = 64       |   32         |    16    |   8 
    // -----------------------------------------------------------
    // warp0  warp1 warp2 warp3 |  warp0 warp1 |   warp0  |  warp0   
    //                          |  warp2 warp3 |   warp1  |  warp1  
    //                          |              |   warp2  |   ...
    //                          |              |   warp3  |  warp7 
    // -----------------------------------------------------------
    int read_row_warps = (BLOCK_ROW_TILES % 16 == 0) ? 
      (BLOCK_ROW_TILES/16):
      (BLOCK_ROW_TILES/16) + 1;
    //int read_col_warps = BLOCK_COL_TILES;
    int readWarpX = warpId / read_row_warps;
    int readWarpY = warpId % read_row_warps;

#if 0
    half *shmem_warp_stream_ptr;
    // This warp's pointer to the C matrix data to copy memory from to shared 
    // memory.
    size_t gmem_idx;
    if(BLOCK_ROW_TILES >= 16){
      shmem_warp_stream_ptr = (half*)&shm[0] + readWarpX * WMMA_M * 
        SHMEM_STRIDE + readWarpY * (WMMA_N * 16);
      gmem_idx = (block_tile_i + readWarpX) * WMMA_M * ldc + 
        (block_tile_j + readWarpY * 16) * WMMA_N;
    }else{ // 8
      shmem_warp_stream_ptr = (half*)&shm[0] + warpId * SHMEM_STRIDE * 
        WMMA_M;
      gmem_idx = (block_tile_i + warpId) * WMMA_M * 
        ldc + block_tile_j * WMMA_N;
    }
#endif

#if 1
    half *shmem_warp_stream_ptr = (BLOCK_ROW_TILES >= 16) ? 
      (half*)&shm[0] + readWarpX * WMMA_M * SHMEM_STRIDE + readWarpY * (WMMA_N *
          16): (half*)&shm[0] + warpId * SHMEM_STRIDE * WMMA_M;

    // This warp's pointer to the C matrix data to copy memory from to shared 
    // memory.
    size_t gmem_idx = (BLOCK_ROW_TILES >= 16) ? 
      ((block_tile_i + readWarpX) * WMMA_M * ldc + (block_tile_j +
        readWarpY * 16) * WMMA_N) :
      (block_tile_i + warpId) * WMMA_M * ldc + block_tile_j * WMMA_N;
#endif



    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.

    if(BLOCK_ROW_TILES >= 16) {
      if(readWarpX < BLOCK_COL_TILES){
        typedef int4 copy_t;
        //#pragma unroll
        for (int i = 0; i < WMMA_K; i++) {
          *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = 
            *((copy_t *)(src_gmem_warp_stream_ptr + ldc * i) + 
                laneId);
        }
      }
    }else{  // BLOCK_ROW_TILES=8
      typedef int2 copy_t;
      //#pragma unroll
      for (int i = 0; i < WMMA_K; i++) {
        *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = 
          *((copy_t *)(src_gmem_warp_stream_ptr + ldc * i) + 
              laneId);
      }
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> 
      acc[WARP_COL_TILES][WARP_ROW_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> 
      c[WARP_COL_TILES][WARP_ROW_TILES];
#if 0
    for(int i = 0;  i < 2; i++)
      for(int j = 0; j < 4; j++)
        wmma::fill_fragment(acc[i][j], 0.0f);
#endif

    // Load the C matrix tiles into fragments from shared memory.
    half *tile_ptr = shmem_warp_tile_ptr - SHMEM_STRIDE  * WMMA_K;
    //#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
      tile_ptr += SHMEM_STRIDE * WMMA_K; 
      //#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        //half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K 
        // + j * WMMA_N;
        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
        tile_ptr += WMMA_N;
      }
    }


    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
        }
      }
    }

    __syncthreads();

    // Assume: CHUNK_K <= 16, so one warp can read one row_line of A/B.
    //   A[BLOCK_COL_TILES][CHUNK_K], B[BLOCK_ROW_TILES][CHUNK_K] 
    //   warp0->A[0],   warp1->A[1], ...,   warpx->A[x], ...,   warpy->B[y]
    //   warp0->A[0+8], warp1->A[1+8], ..., warpx->A[x+8], ..., warpy->B[y+8]
    // Special case when compute c(8, 8)
    //   warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.

    //const half *warp_ptr = (warpId < 4) ? 
    //  (&A[block_tile_i * WMMA_M * lda] + WMMA_M * lda * (warpId % 4) * 2) :
    //  (&B[block_tile_j * WMMA_N * ldb] + WMMA_N * ldb * (warpId % 4) * 2);

    int totalTiles = BLOCK_COL_TILES + BLOCK_ROW_TILES;
    printf("totalTiles = %d \n", totalTiles);

    // Pointer to the tile of A/B.
    half *srcAPtr = &A[block_tile_i * WMMA_M * lda];
    half *srcBPtr = &B[block_tile_j * WMMA_N * ldb];

    // Go through the global K dimension by a fixed step at a time.
    half *srcTilePtr;
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {

      // Step 1. Copy slices of the A and B matrices to shared memory.
#pragma unroll
      for(int nthTile = warpId; nthTile < totalTiles; nthTile +=
          WARPS_PER_BLOCK){

        // Pointer to the tile-pos of the warp.
        srcTilePtr = (nthTile < BLOCK_COL_TILES) ?
          srcAPtr + (nthTile * WMMA_M) * lda + tile_k * WMMA_K:
          srcBPtr + (nthTile - BLOCK_COL_TILES) * WMMA_N * ldb + tile_k *
          WMMA_K;

        // Begin shmem_idx of warp.
        size_t shmemIdx = (nthTile < BLOCK_COL_TILES) ? 
          (WMMA_M * nthTile):
          ((WMMA_M * BLOCK_COL_TILES) + WMMA_N * (nthTile - BLOCK_COL_TILES));

        // shemm_idx of each lane.
        shmemIdx += laneId / CHUNK_COPY_LINE_LANES;

        // Do copy A/B to shmem.
        // -------------------------------------------
        // Tile-A/B-L0  lane0  lane1  lane2  ... lan7
        // Tile-A/B-L1  lane8  lane9  lane10 ... lan15
        // Tile-A/B-L2   ... 
        // Tile-A/B-L3  lane24 lane25 lane26 ... lan31
        // Tile-A/B-L4  lane0 ...
        // Tile-A/B-L5  lane8 ...
        //   ...
        // Tile-A/B-L15
        // -------------------------------------------
#if 1
        int4* lanePtr = (nthTile < BLOCK_COL_TILES)?
          (int4*)(srcTilePtr + (laneId / CHUNK_COPY_LINE_LANES) * lda) + laneId
          % CHUNK_COPY_LINE_LANES:
          (int4*)(srcTilePtr + (laneId / CHUNK_COPY_LINE_LANES) * ldb) + laneId 
          % CHUNK_COPY_LINE_LANES;

#pragma unroll (4)
        for(int i = 0; i < (WMMA_M / CHUNK_COPY_LINES_PER_WARP); i++){
          *((int4*)&shm[shmemIdx * SHMEM_CHUNK_LINE] + laneId % 
              CHUNK_COPY_LINE_LANES) = *lanePtr;

          // Update global pointer and shmem pointer.
          lanePtr = (nthTile < BLOCK_COL_TILES) ?
            (int4*)((half*)lanePtr + lda * CHUNK_COPY_LINES_PER_WARP):
            (int4*)((half*)lanePtr + ldb * CHUNK_COPY_LINES_PER_WARP);
          shmemIdx += CHUNK_COPY_LINES_PER_WARP;
        }
#endif
      }

      __syncthreads();

      // Step 2. Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
          wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
          wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {

          // Load A from shmem to fragment.
          size_t shmem_idx_a = warpX * WARP_COL_TILES * WMMA_M + (i * WMMA_M);
          const half *tilePtr =  &shm[shmem_idx_a * 
            SHMEM_CHUNK_LINE + k_step * WMMA_K];
          wmma::load_matrix_sync(a[i], tilePtr, WMMA_K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {

            if (i == 0) {
              size_t shmem_idx_b = shmem_idx_b_off + warpY * (WARP_ROW_TILES *
                  WMMA_N) + j * WMMA_N; 
              const half *tilePtr = &shm[shmem_idx_b * SHMEM_CHUNK_LINE + 
                k_step * WMMA_K];

              wmma::load_matrix_sync(b[j], tilePtr, WMMA_K * CHUNK_K + 
                  SKEW_HALF);
            }

            wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
          }
        }
      }

      __syncthreads();
    }

    // Step 3. Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        // Uniform, point-wise transformations of ALL fragment elements by ALL 
        // threads in the warp are well-defined even though element indices 
        // within fragment storage are not defined.
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++){
          c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);
        }

        half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j *
          WMMA_N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Step 4. Store the D from shared memory to global memory.
    // Now that shared memory contains all the D tiles, stream them to global 
    // memory.
    //float *dst_gmem_warp_stream_ptr = &D[gmem_idx];
    half *dst_gmem_warp_stream_ptr = &C[gmem_idx];

    if(BLOCK_ROW_TILES >= 16){
      if(readWarpX < BLOCK_COL_TILES){
        typedef int4 copy_t;
#pragma unroll
        for (int i = 0; i < WMMA_K; i++) {
          *((copy_t *)(dst_gmem_warp_stream_ptr + ldc * i) + 
              laneId) = *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) +
                laneId);
        }
      }
    }else{  // BLOCK_ROW_TILES=8
      typedef short4 copy_t;
#pragma unroll
      for (int i = 0; i < WMMA_K; i++) {
        *((copy_t *)(dst_gmem_warp_stream_ptr + ldc * i) + 
            laneId) = *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) +
              laneId); 
      }
    }
    __syncthreads();

  }
}


// Gerneral version
// NT
template<int WmmaM,  int WmmaN, int WmmaK, int 
WarpColTiles, int WarpRowTiles>
__global__ void _gemm_wmma_shm_half_128_16(
    int BlockRowWarps, int BlockColWarps, 
    int WarpsPerBlock, //const int warp_size,
    int ChunkK,
    //bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{
  int SKEW_HALF = 8;

  int  m_tiles = M / WmmaM;
  int  n_tiles = N / WmmaN;
  int  k_tiles = K / WmmaK;

  int BlockRowTiles = BlockRowWarps * WarpRowTiles; 
  int BlockColTiles = BlockColWarps * WarpColTiles; 

  // Info of C/D.
  int ShmemStride = WmmaN * BlockRowTiles;   // C_LINE_LEN of block.
  int ShmemOffset = WmmaN * WarpRowTiles;    // C_LINE_LEN of warp.
  int c_read_lanes = BlockRowTiles * WmmaN / 4;//sizeof(half) / sizeof(int2); 

  // Info of ChunkK- A/B.
  int ChunkLineBytes =  ChunkK * WmmaK * 2; // sizeof(half);      
  //int WarpCopyBytes = 512; //warp_size * sizeof(int4);           
  int ChunkCopyLinesPerWarp = 512 / ChunkLineBytes;  
  int ChunkCopyLineLanes = ChunkLineBytes / 16; //sizeof(int4);   

  // Used for A/B.
  int ShmemChunkLine = ChunkK * WmmaK + SKEW_HALF;
  extern __shared__ half shm[];

  // Warp and lane identification.
  //unsigned int warpY = warpId % BlockRowWarps;
  int tId = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpId = tId / WARP_SIZE;
  //unsigned int laneId = tId % WARP_SIZE;
  unsigned int laneId = tId & WARP_SIZE - 1;
  unsigned int warpX = warpId / BlockRowWarps;
  //unsigned int warpY = warpId % BlockRowWarps;
  unsigned int warpY = warpId & BlockRowWarps - 1;

  // Offset in shared memory from which the B matrix is stored.
  size_t shmem_idx_b_off = BlockColTiles * WmmaM;

  //
  typedef int4 copy4_t;
  typedef int2 copy2_t;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may 
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int grid1D = gridDim.x * gridDim.y;

  unsigned int block_tile_i, block_tile_j, block_pos;
  int run = 1;
  for(int kk = 0; kk < run; kk++){
#pragma unroll
    for(block_pos = blockId; ; block_pos += grid1D) {
      block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) 
        * (BlockColTiles);
      block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

      // Stop when there are no more D matrix tiles to compute in this CTA.
      if (block_tile_i >= m_tiles) {
        break;
      }

      // Step 1. Read C/D.
      // Warps layout.
      // warps_shape(x, y) = (1, 4), (2, 2), (4, 1)
      // warp0 warp1 warp2 warp3    warp0 warp1   warp0 warp1 ... warp7
      // warp4 warp5 warp6 warp7    warp2 warp3
      //                            warp4 warp5
      //                            warp6 warp7
      // This pointer is used to access the C and D matrix tiles this warp 
      half *shmem_warp_tile_ptr = (half*)&shm[0]+ warpX * 
        WarpColTiles * WmmaN * ShmemStride  + warpY * ShmemOffset;

      // This pointer is used to stream the C and D matrices block-wide tile 
      // to and from shared memory. Read from row to cols, warp_read_tile: 
      // Block-tiles-shape:
      // blk_row_tiles = 64       |   32         |    16    |   8 
      // -----------------------------------------------------------
      // warp0  warp1 warp2 warp3 |  warp0 warp1 |   warp0  |  warp0   
      //                          |  warp2 warp3 |   warp1  |  warp1  
      //                          |              |   warp2  |   ...
      //                          |              |   warp3  |  warp7 
      // -----------------------------------------------------------
      // 16 * WMMN * 2 = 512 = 32 * 16
      int warps_read_row = (BlockRowTiles % 16 == 0) ? 
        (BlockRowTiles / 16):
        (BlockRowTiles / 16) + 1;
      int readWarpX = warpId / warps_read_row;
      int readWarpY = warpId % warps_read_row;

      half *shmem_warp_stream_ptr = (BlockRowTiles >= 16) ? (half*)&shm[0] + 
        readWarpX * WmmaM * ShmemStride + readWarpY * (WmmaN *16): 
        (half*)&shm[0] + warpId * ShmemStride * WmmaN;

      // This warp's pointer to the C matrix data to copy memory from to shared 
      // memory.
      size_t gmem_idx = (BlockRowTiles >= 16) ? 
        ((block_tile_i + readWarpX) * WmmaM * ldc + (block_tile_j +
          readWarpY * 16) * WmmaN) :
        (block_tile_i + warpId) * WmmaM * ldc + block_tile_j * WmmaN;
      const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, float> 
        acc[WarpColTiles][WarpRowTiles];
      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, half> 
        c[WarpColTiles][WarpRowTiles];

      if(beta != 0.0f){
        // Step 1.1  Read C from global  mem to shared mem.
        // Stream multiple C tiles to shared memory.
        if(BlockRowTiles >= 16) {
          if(readWarpX < BlockColTiles){
            //#pragma unroll
            for (int i = 0; i < WmmaK; i++) {
              *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
                *((copy4_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
            }
          }
        }else if(BlockRowTiles == 8){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }else{  // BlockRowTiles < 8
          if(warpId < BlockColTiles && laneId < c_read_lanes){
#pragma unroll
            for (int i = 0; i < WmmaK; i++) {
              *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
                *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
            }
          }
        }
        __syncthreads();

        // Step 1.2. Copy C/D from the shm to the fragment.

        // Load the C matrix tiles into fragments from shared memory.
        half *tile_ptr = shmem_warp_tile_ptr - ShmemStride  * WmmaK;
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
          tile_ptr += ShmemStride * WmmaK; 
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            wmma::load_matrix_sync(c[i][j], tile_ptr, ShmemStride, C_LAYOUT);
            tile_ptr += WmmaN;
          }
        }


        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++) {
              acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
            }
          }
        }
        __syncthreads();
      }else{
#pragma unroll
        for(int i = 0;  i < WarpColTiles; i++)
#pragma unroll
          for(int j = 0; j < WarpRowTiles; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);
      }

      // Step 2. Read A/B from global mem to shared mem.
      // Assume: ChunkK <= 16, so one warp can read one row_line of A/B.
      //   A[BlockColTiles][ChunkK], B[BlockRowTiles][ChunkK] 
      //   warp0->A[0],   warp1->A[1], ...,   warpx->A[x], ...,   warpy->B[y]
      //   warp0->A[0+8], warp1->A[1+8], ..., warpx->A[x+8], ..., warpy->B[y+8]
      // Special case when compute c(8, 8)
      //   warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.

      int totalTiles = BlockColTiles + BlockRowTiles;

      // Pointer to the tile of A/B.
      half *srcAPtr = &A[block_tile_i * WmmaM * lda];
      half *srcBPtr = &B[block_tile_j * WmmaN * ldb];

      // Go through the global K dimension by a fixed step at a time.
      half *srcTilePtr;
#pragma unroll
      for (int tile_k = 0; tile_k < k_tiles; tile_k += ChunkK) {

        // Step 1. Copy slices of the A and B matrices to shared memory.
#pragma unroll
        for(int nthTile = warpId; nthTile < totalTiles; nthTile +=
            WarpsPerBlock){

          // Pointer to the tile-pos of the warp.
          srcTilePtr = (nthTile < BlockColTiles) ?
            srcAPtr + (nthTile * WmmaM) * lda + tile_k * WmmaK:
            srcBPtr + (nthTile - BlockColTiles) * WmmaN * ldb + tile_k *
            WmmaK;

          // Begin shmem_idx of warp.
          size_t shmemIdx = (nthTile < BlockColTiles) ? 
            (WmmaM * nthTile):
            ((WmmaM * BlockColTiles) + WmmaN * (nthTile - BlockColTiles));

          // shemm_idx of each lane.
          shmemIdx += laneId / ChunkCopyLineLanes;

          // Do copy A/B to shmem.
          // -------------------------------------------
          // Tile-A/B-L0  lane0  lane1  lane2  ... lan7
          // Tile-A/B-L1  lane8  lane9  lane10 ... lan15
          // Tile-A/B-L2   ... 
          // Tile-A/B-L3  lane24 lane25 lane26 ... lan31
          // Tile-A/B-L4  lane0 ...
          // Tile-A/B-L5  lane8 ...
          //   ...
          // Tile-A/B-L15
          // -------------------------------------------
#if 1
          int4* lanePtr = (nthTile < BlockColTiles)?
            (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) * lda) + laneId
            % ChunkCopyLineLanes:
            (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) * ldb) + laneId 
            % ChunkCopyLineLanes;

#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (nthTile < BlockColTiles) ?
              (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp):
              (int4*)((half*)lanePtr + ldb * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }
#endif
        }

        __syncthreads();

        // Step 2. Compute a grid of C matrix tiles in each warp.
#pragma unroll
        for (int k_step = 0; k_step < ChunkK; k_step++) {
          wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
            wmma::row_major> a[WarpColTiles];
          wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, 
            wmma::col_major> b[WarpRowTiles];

#pragma unroll
          for (int i = 0; i < WarpColTiles; i++) {

            // Load A from shmem to fragment.
            size_t shmem_idx_a = warpX * WarpColTiles * WmmaM + (i * WmmaM);
            const half *tilePtr =  &shm[shmem_idx_a * 
              ShmemChunkLine + k_step * WmmaK];
            wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);

#pragma unroll
            for (int j = 0; j < WarpRowTiles; j++) {

              if (i == 0) {
                size_t shmem_idx_b = shmem_idx_b_off + warpY * (WarpRowTiles *
                    WmmaN) + j * WmmaN; 
                const half *tilePtr = &shm[shmem_idx_b * ShmemChunkLine + 
                  k_step * WmmaK];

                wmma::load_matrix_sync(b[j], tilePtr, WmmaK * ChunkK + 
                    SKEW_HALF);
              }

              wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
            }
          }
        }

        __syncthreads();
      }

      // Step 3. Store the D fragments to shared memory.
#pragma unroll
      for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
        for (int j = 0; j < WarpRowTiles; j++) {
          // Uniform, point-wise transformations of ALL fragment elements by ALL 
          // threads in the warp are well-defined even though element indices 
          // within fragment storage are not defined.
#pragma unroll
          for (int t = 0; t < c[i][j].num_elements; t++){
            c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);
          }

          half *tile_ptr = shmem_warp_tile_ptr + i * ShmemStride * WmmaK + j *
            WmmaN;
          wmma::store_matrix_sync(tile_ptr, c[i][j], ShmemStride, C_LAYOUT);
        }
      }

      __syncthreads();

      // Step 4. Store the D from shared memory to global memory.
      // Now that shared memory contains all the D tiles, stream them to global 
      // memory.
      //float *dst_gmem_warp_stream_ptr = &D[gmem_idx];
      half *dst_gmem_warp_stream_ptr = &C[gmem_idx];

      if(BlockRowTiles >= 16){
        if(readWarpX < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){  // BlockRowTiles=8
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId); 
        }
      }else{  // BlockRowTiles=8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK && laneId < c_read_lanes; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId); 
          }
        }
      }
      __syncthreads();

    }
  }

}


// New than _gemm_wmma_shm_half_config() 
template<int WmmaM,  int WmmaN, int WmmaK, int 
WarpColTiles, int WarpRowTiles>
__global__ void _gemm_wmma_shm_r_opt(
    int BlockRowWarps, int BlockColWarps, 
    int WarpsPerBlock, //const int warp_size,
    int ChunkK,
    //bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{
  const int SKEW_HALF = 8;

  const int  m_tiles = 8; //M / WmmaM;
  const int  n_tiles = 192; //N / WmmaN;
  const int  k_tiles = 64; //K / WmmaK;

  const int BlockRowTiles = 8; //BlockRowWarps * WarpRowTiles; 
  const int BlockColTiles = 8; //BlockColWarps * WarpColTiles; 

  // Info of C/D.
  int ShmemStride = WmmaN * BlockRowTiles;   // C_LINE_LEN of block.
  int ShmemOffset = WmmaN * WarpRowTiles;    // C_LINE_LEN of warp.
  int c_read_lanes = BlockRowTiles * WmmaN / 4;//sizeof(half) / sizeof(int2); 

  // Info of ChunkK- A/B.
  int ChunkLineBytes =  ChunkK * WmmaK * 2; // sizeof(half);      
  //int WarpCopyBytes = 512; //warp_size * sizeof(int4);           
  int ChunkCopyLinesPerWarp = 512 / ChunkLineBytes;  
  int ChunkCopyLineLanes = ChunkLineBytes / 16; //sizeof(int4);   

  // Used for A/B.
  int ShmemChunkLine = ChunkK * WmmaK + SKEW_HALF;
  extern __shared__ half shm[];

  // Warp and lane identification.
  //unsigned int warpY = warpId % BlockRowWarps;
  //unsigned int laneId = tId % WARP_SIZE;
  //unsigned int warpY = warpId % BlockRowWarps;
  int tId = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpId = tId / WARP_SIZE;
  unsigned int laneId = tId & WARP_SIZE - 1;
  unsigned int warpX = warpId / BlockRowWarps;
  unsigned int warpY = warpId & BlockRowWarps - 1;

  // Offset in shared memory from which the B matrix is stored.
  size_t shmem_idx_b_off = BlockColTiles * WmmaM;

  //
  typedef int4 copy4_t;
  typedef int2 copy2_t;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may 
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int grid1D = gridDim.x * gridDim.y;

  unsigned int block_tile_i, block_tile_j, block_pos;
  int run = 1;
  for(int kk = 0; kk < run; kk++){
#pragma unroll
    for(block_pos = blockId; ; block_pos += grid1D) {
      block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) 
        * (BlockColTiles);
      block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

      // Stop when there are no more D matrix tiles to compute in this CTA.
      if (block_tile_i >= m_tiles) {
        break;
      }

      // Step 1. Read C/D.
      // Warps layout.
      // warps_shape(x, y) = (1, 4), (2, 2), (4, 1)
      // warp0 warp1 warp2 warp3    warp0 warp1   warp0 warp1 ... warp7
      // warp4 warp5 warp6 warp7    warp2 warp3
      //                            warp4 warp5
      //                            warp6 warp7
      // This pointer is used to access the C and D matrix tiles this warp 
      half *shmem_warp_tile_ptr = (half*)&shm[0]+ warpX * 
        WarpColTiles * WmmaN * ShmemStride  + warpY * ShmemOffset;

      // This pointer is used to stream the C and D matrices block-wide tile 
      // to and from shared memory. Read from row to cols, warp_read_tile: 
      // Block-tiles-shape:
      // blk_row_tiles = 64       |   32         |    16    |   8 
      // -----------------------------------------------------------
      // warp0  warp1 warp2 warp3 |  warp0 warp1 |   warp0  |  warp0   
      //                          |  warp2 warp3 |   warp1  |  warp1  
      //                          |              |   warp2  |   ...
      //                          |              |   warp3  |  warp7 
      // -----------------------------------------------------------
      // 16 * WMMN * 2 = 512 = 32 * 16
      //int warps_read_row = 1;
      //int readWarpX = warpId; // / warps_read_row;
      //int readWarpY = 0; // warpId % warps_read_row;

      half *shmem_warp_stream_ptr = (half*)&shm[0] + warpId * ShmemStride *
        WmmaN;

      // This warp's pointer to the C matrix data to copy memory from to shared 
      // memory.
      size_t gmem_idx = (block_tile_i + warpId) * WmmaM * ldc + block_tile_j
        * WmmaN;

      const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

      // Step 1.1  Read C from global  mem to shared mem.
      // Stream multiple C tiles to shared memory.
      if(BlockRowTiles == 8){
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
            *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles && laneId < c_read_lanes){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }
      __syncthreads();

      // Step 1.2. Copy C/D from the shm to the fragment.
      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, float> 
        acc[WarpColTiles][WarpRowTiles];
      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, half> 
        c[WarpColTiles][WarpRowTiles];
#if 0
      for(int i = 0;  i < 2; i++)
        for(int j = 0; j < 4; j++)
          wmma::fill_fragment(acc[i][j], 0.0f);
#endif

      // Load the C matrix tiles into fragments from shared memory.
      half *tile_ptr = shmem_warp_tile_ptr - ShmemStride  * WmmaK;
#pragma unroll
      for (int i = 0; i < WarpColTiles; i++) {
        tile_ptr += ShmemStride * WmmaK; 
#pragma unroll
        for (int j = 0; j < WarpRowTiles; j++) {
          wmma::load_matrix_sync(c[i][j], tile_ptr, ShmemStride, C_LAYOUT);
          tile_ptr += WmmaN;
        }
      }


      // Scale the C matrix.
#pragma unroll
      for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
        for (int j = 0; j < WarpRowTiles; j++) {
#pragma unroll
          for (int t = 0; t < c[i][j].num_elements; t++) {
            acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
          }
        }
      }

      __syncthreads();

      // Step 2. Read A/B from global mem to shared mem.
      // Assume: ChunkK <= 16, so one warp can read one row_line of A/B.
      //   A[BlockColTiles][ChunkK], B[BlockRowTiles][ChunkK] 
      //   warp0->A[0],   warp1->A[1], ...,   warpx->A[x], ...,   warpy->B[y]
      //   warp0->A[0+8], warp1->A[1+8], ..., warpx->A[x+8], ..., warpy->B[y+8]
      // Special case when compute c(8, 8)
      //   warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.

      int totalTiles = BlockColTiles + BlockRowTiles;

      // Pointer to the tile of A/B.
      half *srcAPtr = &A[block_tile_i * WmmaM * lda];
      half *srcBPtr = &B[block_tile_j * WmmaN * ldb];

      // Go through the global K dimension by a fixed step at a time.
      half *srcTilePtr;
#pragma unroll
      for (int tile_k = 0; tile_k < k_tiles; tile_k += ChunkK) {

        // Step 1. Copy slices of the A and B matrices to shared memory.
#pragma unroll
        for(int nthTile = warpId; nthTile < totalTiles; nthTile +=
            WarpsPerBlock){

          // Pointer to the tile-pos of the warp.
          srcTilePtr = (nthTile < BlockColTiles) ?
            srcAPtr + (nthTile * WmmaM) * lda + tile_k * WmmaK:
            srcBPtr + (nthTile - BlockColTiles) * WmmaN * ldb + tile_k *
            WmmaK;

          // Begin shmem_idx of warp.
          size_t shmemIdx = (nthTile < BlockColTiles) ? 
            (WmmaM * nthTile):
            ((WmmaM * BlockColTiles) + WmmaN * (nthTile - BlockColTiles));

          // shemm_idx of each lane.
          shmemIdx += laneId / ChunkCopyLineLanes;

          // Do copy A/B to shmem.
          // -------------------------------------------
          // Tile-A/B-L0  lane0  lane1  lane2  ... lan7
          // Tile-A/B-L1  lane8  lane9  lane10 ... lan15
          // Tile-A/B-L2   ... 
          // Tile-A/B-L3  lane24 lane25 lane26 ... lan31
          // Tile-A/B-L4  lane0 ...
          // Tile-A/B-L5  lane8 ...
          //   ...
          // Tile-A/B-L15
          // -------------------------------------------
#if 1
          int4* lanePtr = (nthTile < BlockColTiles)?
            (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) * lda) + laneId
            % ChunkCopyLineLanes:
            (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) * ldb) + laneId 
            % ChunkCopyLineLanes;

#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (nthTile < BlockColTiles) ?
              (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp):
              (int4*)((half*)lanePtr + ldb * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }
#endif
        }

        __syncthreads();

        // Step 2. Compute a grid of C matrix tiles in each warp.
#pragma unroll
        for (int k_step = 0; k_step < ChunkK; k_step++) {
          wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
            wmma::row_major> a[WarpColTiles];
          wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, 
            wmma::col_major> b[WarpRowTiles];

#pragma unroll
          for (int i = 0; i < WarpColTiles; i++) {

            // Load A from shmem to fragment.
            size_t shmem_idx_a = warpX * WarpColTiles * WmmaM + (i * WmmaM);
            const half *tilePtr =  &shm[shmem_idx_a * 
              ShmemChunkLine + k_step * WmmaK];
            wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);

#pragma unroll
            for (int j = 0; j < WarpRowTiles; j++) {

              if (i == 0) {
                size_t shmem_idx_b = shmem_idx_b_off + warpY * (WarpRowTiles *
                    WmmaN) + j * WmmaN; 
                const half *tilePtr = &shm[shmem_idx_b * ShmemChunkLine + 
                  k_step * WmmaK];

                wmma::load_matrix_sync(b[j], tilePtr, WmmaK * ChunkK + 
                    SKEW_HALF);
              }

              wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
            }
          }
        }

        __syncthreads();
      }

      // Step 3. Store the D fragments to shared memory.
#pragma unroll
      for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
        for (int j = 0; j < WarpRowTiles; j++) {
          // Uniform, point-wise transformations of ALL fragment elements by ALL 
          // threads in the warp are well-defined even though element indices 
          // within fragment storage are not defined.
#pragma unroll
          for (int t = 0; t < c[i][j].num_elements; t++){
            c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);
          }

          half *tile_ptr = shmem_warp_tile_ptr + i * ShmemStride * WmmaK + j *
            WmmaN;
          wmma::store_matrix_sync(tile_ptr, c[i][j], ShmemStride, C_LAYOUT);
        }
      }

      __syncthreads();

      // Step 4. Store the D from shared memory to global memory.
      // Now that shared memory contains all the D tiles, stream them to global 
      // memory.
      half *dst_gmem_warp_stream_ptr = &C[gmem_idx];

      if(BlockRowTiles == 8){  // BlockRowTiles=8
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId); 
        }
      }else{  // BlockRowTiles=8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK && laneId < c_read_lanes; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId); 
          }
        }
      }
      __syncthreads();

    }
  }

}


#if 0
// NT. 
template<int WmmaM,  int WmmaN, int WmmaK, int 
  WarpColTiles, int WarpRowTiles, int ChunkCol>
  __global__ void __launch_bounds__(256, 1)
_gemm_wmma_shm_persistent(
    int BlockRowWarps, int BlockColWarps, 
    int WarpsPerBlock,
    int ChunkK,
    int M, int N, int K,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{
  int SKEW_HALF = 8;

  int  m_tiles = M / WmmaM;
  int  n_tiles = N / WmmaN;
  int  k_tiles = K / WmmaK;

  int BlockRowTiles = BlockRowWarps * WarpRowTiles; 
  int BlockColTiles = BlockColWarps * WarpColTiles; 

  // Info of C/D.
  int ShmemStride = WmmaN * BlockRowTiles;   // C_LINE_LEN of block.
  int ShmemOffset = WmmaN * WarpRowTiles;    // C_LINE_LEN of warp.
  int c_read_lanes = BlockRowTiles * WmmaN / 4;//sizeof(half) / sizeof(int2); 

  // Info of ChunkK- A/B.
  int ChunkLineBytes =  ChunkK * WmmaK * 2; // sizeof(half);      
  int WarpCopyBytes = 512; //warp_size * sizeof(int4);           
  int ChunkCopyLinesPerWarp = 512 / ChunkLineBytes;  
  int ChunkCopyLineLanes = ChunkLineBytes / 16; //sizeof(int4);   

  // Used for A/B.
  int ShmemChunkLine = ChunkK * WmmaK + SKEW_HALF;
  extern __shared__ half shm[];

  // Offset in shared memory from which the B matrix is stored.
  //size_t shmem_idx_b_off = BlockColTiles * WmmaM;

  typedef int4 copy4_t;
  typedef int2 copy2_t;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may 
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int grid1D = gridDim.x * gridDim.y;

  int tId = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpId = tId / WARP_SIZE;
  unsigned int laneId = tId & WARP_SIZE - 1;
  unsigned int warpX = warpId / BlockRowWarps;
  unsigned int warpY = warpId & BlockRowWarps - 1;

  unsigned int block_tile_i, block_tile_j, block_pos;

  // Step 1. Init: copy B to registers.
  bool loadB = false;
  wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, 
    wmma::col_major> b[WarpRowTiles][ChunkCol];

#pragma unroll
  for(block_pos = blockId; ; block_pos += grid1D) {
    block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) * (BlockColTiles);
    block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

    if (loadB || block_tile_i >= m_tiles) 
      break;
    loadB = true;

    half *srcBPtr = &B[block_tile_j * WmmaN * ldb];
    half *srcTilePtr;

    // k_tiles % ChunkK == 0.
#pragma unroll
    for (int tile_k = 0; tile_k < k_tiles; tile_k += ChunkK) {
#pragma unroll
      for(int nthTile = warpId * WarpRowTiles, nth =0; nth < WarpRowTiles; 
          nth++, nthTile++){

        // Begin shmem_idx of warp.
        size_t shmemIdx =  warpId * WmmaN; 

        if(warpX == 0){

          // Step 1.1: Copy B from gmem to shmem.
          // Pointer to the tile-pos of the warp.
          srcTilePtr = srcBPtr + nthTile * WmmaN * ldb + tile_k * WmmaK;

          // shemm_idx/lanePtr of each lane.
          int4* lanePtr = (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) * ldb) 
            + laneId % ChunkCopyLineLanes;
          shmemIdx += laneId / ChunkCopyLineLanes;

#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (int4*)((half*)lanePtr + ldb * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }

        }
        __syncthreads();

        // Copy B from shmem to registers.
        if(warpX == 0){
          shmemIdx = warpId * WmmaN; 
          for(int k_step = 0; k_step < ChunkK; k_step++){
            const half *tilePtr = &shm[shmemIdx * ShmemChunkLine + 
              k_step * WmmaK];
            wmma::load_matrix_sync(b[nth][tile_k + k_step], tilePtr, WmmaK * 
                ChunkK + SKEW_HALF);
          }
        }
        __syncthreads();
      }
    } // ChunkK
  }

  // Do load(A, C) and compute(C = A * B).
  // for: blk_tiles(blkTilesX, blkTilesY) warp_tiles(warpTilesX, warpTilesY)
  // assume: warpTilesX == blkTilesX
  //           |     Y     |              Y                       Y
  //  X=0,1,...|           | 
  // ----------|-----------|-----------------------|---------------------|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|

  for(int run = 0; run < 10000; run++)
  {
#pragma unroll
    for(block_pos = blockId; ; block_pos += grid1D) {
      block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) 
        * (BlockColTiles);
      block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

      // Stop when there are no more D matrix tiles to compute in this CTA.
      if (block_tile_i >= m_tiles) {
        break;
      }

      // Step 1. Load C.
      // BlkTiles(x, y): (8, 4), (8, 8), 1<=x<=8, y = 4, 8, ...
      //               ----> readWarpY 
      // BlkRowTiles = |0     15|16    31|
      //               |--------|--------|
      //     readWarpX |  warp0 | warp1  |
      //               |  warp2 | warp3  |
      //               |  warp4 | warp5  |
      //               |  ...   |        |
      //               |  warp6 | warp7  |
      //               |--------|--------|

      // This warp's pointer  access the C and D shm tile.
      half *shmem_warp_tile_ptr = (half*)&shm[0] + warpX * 
        WarpColTiles * WmmaN * ShmemStride  + warpY * ShmemOffset;

      // Copy C from gmem to shmem.
      // This idx point to the tile the warp read.
      // 16 * WMMN * 2 = 512 = 32 * 16
      int warpsReadRow = (BlockRowTiles % 16 == 0) ? 
        (BlockRowTiles / 16):
        (BlockRowTiles / 16) + 1;
      int readWarpX = warpId / warpsReadRow;
      int readWarpY = warpId % warpsReadRow;

      // This warp's pointer to the shared memory.
      half *shmem_warp_stream_ptr = (BlockRowTiles >= 16) ? (half*)&shm[0] + 
        readWarpX * WmmaM * ShmemStride + readWarpY * (WmmaN * 16): 
        (half*)&shm[0] + warpId * ShmemStride * WmmaN;

      // This warp's pointer to the C matrix that is copied to shared memory.
      size_t gmem_idx = (BlockRowTiles >= 16) ? 
        ((block_tile_i + readWarpX) * WmmaM * ldc + (block_tile_j +
          readWarpY * 16) * WmmaN) :
        (block_tile_i + warpId) * WmmaM * ldc + block_tile_j * WmmaN;
      const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

      // Step 1.1  Read C from global  mem to shared mem.
      // Stream multiple C tiles to shared memory.
      if(BlockRowTiles >= 16) {
        if(readWarpX < BlockColTiles){
          //#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy4_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
            *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles && laneId < c_read_lanes){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }
      __syncthreads();

      // Step 1.2. Copy C/D from the shm to the fragment.
      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, float> 
        acc[WarpColTiles][WarpRowTiles];

      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, half> 
        c[WarpColTiles][WarpRowTiles];
#if 1
      for(int i = 0;  i < 2; i++)
        for(int j = 0; j < 4; j++)
          wmma::fill_fragment(acc[i][j], 0.0f);
#endif

      // Load the C matrix tiles into fragments from shared memory.
      // Part of warps do this.
      if(warpX < BlockColWarps){
        half *tile_ptr = shmem_warp_tile_ptr - ShmemStride  * WmmaK;
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
          tile_ptr += ShmemStride * WmmaK; 
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            wmma::load_matrix_sync(c[i][j], tile_ptr, ShmemStride, C_LAYOUT);
            tile_ptr += WmmaN;
          }
        }

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++) {
              acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
            }
          }
        }
      }
      __syncthreads();

      // Step 2. Read A from global mem to shared mem.
      // Assume: ChunkK <= 16, so one warp can read one row_line of A/B.
      //   A[BlockColTiles][ChunkK], B[BlockRowTiles][ChunkK] 
      //   warp0->A[0],   warp1->A[1], ...,   warpx->A[x], ...,   warpy->B[y]
      //   warp0->A[0+8], warp1->A[1+8], ..., warpx->A[x+8], ..., warpy->B[y+8]

      // Pointer to the tile of A/B.
      half *srcAPtr = &A[block_tile_i * WmmaM * lda];
      half *srcTilePtr;
#if 1
#pragma unroll
      for (int tile_k = 0; tile_k < k_tiles; tile_k += ChunkK) {

        // Step 1. Copy slices of the A matrices to shared memory.
#pragma unroll
        for(int nthTile = warpId; nthTile < BlockColTiles; nthTile +=
            WarpsPerBlock){

          // Pointer to the tile-pos of the warp.
          srcTilePtr = srcAPtr + (nthTile * WmmaM) * lda + tile_k * WmmaK;

          // Begin shmem_idx of warp.
          size_t shmemIdx = WmmaM * nthTile;

          // shemm_idx of each lane.
          shmemIdx += laneId / ChunkCopyLineLanes;

          // Do copy A/B to shmem.
          // -------------------------------------------
          // Tile-A/B-L0  lane0  lane1  lane2  ... lan7
          // Tile-A/B-L1  lane8  lane9  lane10 ... lan15
          // Tile-A/B-L2   ... 
          // Tile-A/B-L3  lane24 lane25 lane26 ... lan31
          // Tile-A/B-L4  lane0 ...
          // Tile-A/B-L5  lane8 ...
          //   ...
          // Tile-A/B-L15
          // -------------------------------------------
          int4* lanePtr =  (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) *
              lda) + laneId % ChunkCopyLineLanes;

#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }
        }

        __syncthreads();
#if 0
        if(blockIdx.x == 0 && tId == 0){
          //printf("tile_k = %d, \n", tile_k);
          for(int i = 0; i < 64; i++){
            printf("[%d] = %f \n", i, __half2float(shm[i]));
          }
        }
#endif

        // Step 2. Compute a grid of C matrix tiles in each warp.
        if(warpX < BlockColWarps){
#pragma unroll
          for (int k_step = 0; k_step < ChunkK; k_step++) {
            wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
              wmma::row_major> a[WarpColTiles];
#pragma unroll
            for (int i = 0; i < WarpColTiles; i++) {
              // Load A from shmem to fragment.
              size_t shmem_idx_a = warpX * WarpColTiles * WmmaM + (i * WmmaM);
              const half *tilePtr =  &shm[shmem_idx_a * ShmemChunkLine + k_step 
                * WmmaK];
              wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);
#pragma unroll
              for (int j = 0; j < WarpRowTiles; j++) {
                wmma::mma_sync(acc[i][j], a[i], b[j][tile_k + k_step], acc[i][j]);
              }
            }
          }
        }

        __syncthreads();
      }
#endif

      // Step 3. Store the D fragments to shared memory.
      if(warpX < BlockColWarps){
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            // Uniform, point-wise transformations of ALL fragment elements by ALL 
            // threads in the warp are well-defined even though element indices 
            // within fragment storage are not defined.
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++){
              c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);
            }

            half *tile_ptr = shmem_warp_tile_ptr + i * ShmemStride * WmmaM + j *
              WmmaN;
            wmma::store_matrix_sync(tile_ptr, c[i][j], ShmemStride, C_LAYOUT);
          }
        }
      }

      __syncthreads();

      // Step 4. Store the D from shared memory to global memory.
      // Now that shared memory contains all the D tiles, stream them to global 
      // memory.

      half *dst_gmem_warp_stream_ptr = &C[gmem_idx];
      if(BlockRowTiles >= 16){
        if(readWarpX < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){  // BlockRowTiles=8
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId); 
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK && laneId < c_read_lanes; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId); 
          }
        }
      }


      __syncthreads();

    }
  }
}
#endif

// NT. 
template<int WmmaM,  int WmmaN, int WmmaK, int 
  WarpColTiles, int WarpRowTiles, int ChunkCol, int ChunkK>
  __global__ void __launch_bounds__(256, 1)
_gemm_wmma_shm_persistent(
    int BlockRowWarps, int BlockColWarps, 
    int WarpsPerBlock,
    int M, int N, int K,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{
  int SKEW_HALF = 8;

  int  m_tiles = M / WmmaM;
  int  n_tiles = N / WmmaN;

  int BlockRowTiles = BlockRowWarps * WarpRowTiles; 
  int BlockColTiles = BlockColWarps * WarpColTiles; 

  // Info of C/D.
  int ShmemStride = WmmaN * BlockRowTiles;   // C_LINE_LEN of block.
  int ShmemOffset = WmmaN * WarpRowTiles;    // C_LINE_LEN of warp.
  int c_read_lanes = BlockRowTiles * WmmaN / 4;//sizeof(half) / sizeof(int2); 

  // Info of ChunkK- A/B.
  int ChunkCopyLinesPerWarp = 512 / (ChunkK * WmmaK * 2);  
  int ChunkCopyLineLanes = (ChunkK * WmmaK * 2) / 16; // sizeof(int4);   


  // Used for A/B.
  int ShmemChunkLine = ChunkK * WmmaK + SKEW_HALF;
  extern __shared__ half shm[];

  // Offset in shared memory from which the B matrix is stored.
  //size_t shmem_idx_b_off = BlockColTiles * WmmaM;

  typedef int4 copy4_t;
  typedef int2 copy2_t;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may 
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int grid1D = gridDim.x * gridDim.y;

  int tId = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpId = tId / WARP_SIZE;
  unsigned int laneId = tId & WARP_SIZE - 1;
  unsigned int warpX = warpId / BlockRowWarps;
  unsigned int warpY = warpId & BlockRowWarps - 1; // % BlockRowWarps.

  unsigned int block_tile_i, block_tile_j, block_pos;

  // Step 1. Init: copy B to registers.
  bool loadB = false;

  wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, 
    wmma::col_major> b[WarpRowTiles][ChunkCol];
#if 1
  for(block_pos = blockId; ; block_pos += grid1D) {
    block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) * (BlockColTiles);
    block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

    if (block_tile_i >= m_tiles) 
      break;

    half *srcBPtr = &B[block_tile_j * WmmaN * ldb];
    half *srcWarpPtr;
    size_t shmemIdx;  

#pragma unroll
    for (int tile_k = 0; tile_k < ChunkCol; tile_k += ChunkK) {
#pragma unroll
      for(int nthTile = warpId * WarpRowTiles, nth =0; nth < WarpRowTiles; 
          nth++, nthTile++){

        // Step 1.1: Copy B from gmem to shmem.
        // Begin shmem_idx of warp.
        shmemIdx =  warpId * WmmaN; 
        srcWarpPtr = srcBPtr + nthTile * WmmaN * ldb + tile_k * WmmaK;

        // Pointer to the tile-pos of the warp.
        int4* lanePtr = (int4*)(srcWarpPtr + (laneId / ChunkCopyLineLanes) * ldb) 
          + laneId % ChunkCopyLineLanes;
        // Begin shemm_idx of lane.
        shmemIdx += laneId / ChunkCopyLineLanes;
#if 1
#pragma unroll 
        for(int i = 0; i < (WmmaN / ChunkCopyLinesPerWarp); i++){
          *((int4*)(&shm[shmemIdx * ShmemChunkLine]) + laneId % 
              ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

          // Update global pointer and shmem pointer.
          lanePtr = (int4*)((half*)lanePtr + ldb * ChunkCopyLinesPerWarp);
          shmemIdx += ChunkCopyLinesPerWarp;
        }
#endif
        __syncthreads();

        shmemIdx = warpId * WmmaN;  // init.
#pragma unroll
        for(int k_step = 0; k_step < ChunkK; k_step++){

          const half *tilePtr = &shm[shmemIdx * ShmemChunkLine + 
            k_step * WmmaK];
          wmma::load_matrix_sync(b[nth][tile_k + k_step], tilePtr, WmmaK * 
              ChunkK + SKEW_HALF);
        }
        __syncthreads();

      }
    }
  }
#endif


  // Do load(A, C) and compute(C = A * B).
  // for: blk_tiles(blkTilesX, blkTilesY) warp_tiles(warpTilesX, warpTilesY)
  // assume: warpTilesX == blkTilesX
  //           |     Y     |              Y                       Y
  //  X=0,1,...|           | 
  // ----------|-----------|-----------------------|---------------------|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|

#if 1
  int runtimes = 1; //102720;
  for(int run = 0; run < runtimes; run++){
#pragma unroll
    for(block_pos = blockId; ; block_pos += grid1D) {
      block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) 
        * (BlockColTiles);
      block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

      // Stop when there are no more D matrix tiles to compute in this CTA.
      if (block_tile_i >= m_tiles) 
        break;

      // Step 1. Load C.
      // BlkTiles(x, y): (8, 4), (8, 8), 1<=x<=8, y = 4, 8, ...
      //               ----> readWarpY 
      // BlkRowTiles = |0     15|16    31|
      //               |--------|--------|
      //     readWarpX |  warp0 | warp1  |
      //               |  warp2 | warp3  |
      //               |  warp4 | warp5  |
      //               |  ...   |        |
      //               |  warp6 | warp7  |
      //               |--------|--------|

      // This warp's pointer  access the C and D shm tile.
      half *shmem_warp_tile_ptr = (half*)&shm[0] + warpX * 
        WarpColTiles * WmmaN * ShmemStride  + warpY * ShmemOffset;

      // Copy C from gmem to shmem.
      // This idx point to the tile the warp read.
      // 16 * WMMN * 2 = 512 = 32 * 16
      int warpsReadRow = (BlockRowTiles % 16 == 0) ? 
        (BlockRowTiles / 16):
        (BlockRowTiles / 16) + 1;
      int readWarpX = warpId / warpsReadRow;
      int readWarpY = warpId % warpsReadRow;


      // This warp's pointer to the shared memory.
      half *shmem_warp_stream_ptr = (BlockRowTiles >= 16) ? (half*)&shm[0] + 
        readWarpX * WmmaM * ShmemStride + readWarpY * (WmmaN * 16): 
        (half*)&shm[0] + warpId * ShmemStride * WmmaN;

      // This warp's pointer to the C matrix that is copied to shared memory.
      size_t gmem_idx = (BlockRowTiles >= 16) ? 
        ((block_tile_i + readWarpX) * WmmaM * ldc + (block_tile_j +
          readWarpY * 16) * WmmaN) :
        (block_tile_i + warpId) * WmmaM * ldc + block_tile_j * WmmaN;
      const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, float> 
        acc[WarpColTiles][WarpRowTiles];
      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, half> 
        c[WarpColTiles][WarpRowTiles];

#if 0
      // Step 1.1  Read C from global  mem to shared mem.
      // Stream multiple C tiles to shared memory.
      if(BlockRowTiles >= 16) {
        if(readWarpX < BlockColTiles){
          //#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy4_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
            *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles && laneId < c_read_lanes){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }
      __syncthreads();

      // Step 1.2. Copy C/D from the shm to the fragment.
#if 1
#pragma unroll
      for(int i = 0;  i < WarpColTiles; i++){
#pragma unroll
        for(int j = 0; j < WarpRowTiles; j++){
          wmma::fill_fragment(acc[i][j], 0.0f);
        }
      }
#endif

      // Load the C matrix tiles into fragments from shared memory.
      // Part of warps do this.
      //if(warpX < BlockColWarps)
      {
        half *tile_ptr = shmem_warp_tile_ptr - ShmemStride  * WmmaK;
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
          tile_ptr += ShmemStride * WmmaK; 
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            wmma::load_matrix_sync(c[i][j], tile_ptr, ShmemStride, C_LAYOUT);
            tile_ptr += WmmaN;
          }
        }

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++) {
              acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
            }
          }
        }
      }
      __syncthreads();
#endif

#if 1
      for(int i = 0;  i < WarpColTiles; i++)
        for(int j = 0; j < WarpRowTiles; j++)
          wmma::fill_fragment(acc[i][j], 0.0f);
      //__syncthreads();
#endif


      // Step 2. Read A from global mem to shared mem.
      // Assume: ChunkK <= 16, so one warp can read one row_line of A/B.
      //   A[BlockColTiles][ChunkK], B[BlockRowTiles][ChunkK] 
      //   warp0->A[0],   warp1->A[1], ...,   warpx->A[x], ...,   warpy->B[y]
      //   warp0->A[0+8], warp1->A[1+8], ..., warpx->A[x+8], ..., warpy->B[y+8]

      // Pointer to the tile of A/B.
      half *srcAPtr = &A[block_tile_i * WmmaM * lda];
      half *srcTilePtr;

#if 1
#pragma unroll
      for (int tile_k = 0; tile_k < ChunkCol; tile_k += ChunkK){

        // Step 1. Copy slices of the A matrices to shared memory.
#pragma unroll
        for(int nthTile = warpId; nthTile < BlockColTiles; nthTile +=
            WarpsPerBlock){

          // Pointer to the tile-pos of the warp.
          srcTilePtr = srcAPtr + (nthTile * WmmaM) * lda + tile_k * WmmaK;

          // Begin shmem_idx of warp.
          size_t shmemIdx = WmmaM * nthTile;

          // shemm_idx of each lane.
          shmemIdx += laneId / ChunkCopyLineLanes;

          // Do copy A/B to shmem.
          int4* lanePtr =  (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) *
              lda) + laneId % ChunkCopyLineLanes;
#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }
        }

        __syncthreads();

        // Step 2. Compute a grid of C matrix tiles in each warp.
        //if(warpX < BlockColWarps)
        {
          wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
            wmma::row_major> a[WarpColTiles];
#pragma unroll
          for (int k_step = 0; k_step < ChunkK; k_step++) {
#pragma unroll
            for (int i = 0; i < WarpColTiles; i++) {
              // Load A from shmem to fragment.
              size_t shmem_idx_a = warpX * WarpColTiles * WmmaM + (i * WmmaM);
              const half *tilePtr =  &shm[shmem_idx_a * ShmemChunkLine + k_step 
                * WmmaK];
              wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);
#pragma unroll
              for (int j = 0; j < WarpRowTiles; j++) {
                wmma::mma_sync(acc[i][j], a[i], b[j][tile_k + k_step], acc[i][j]);
              }
            }
          }
        }

        __syncthreads();
      }


      // Step 3. Store the D fragments to shared memory.
      //if(warpX < BlockColWarps)
      {
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            // Uniform, point-wise transformations of ALL fragment elements by ALL 
            // threads in the warp are well-defined even though element indices 
            // within fragment storage are not defined.
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++){
              c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);
            }

            half *tile_ptr = shmem_warp_tile_ptr + i * ShmemStride * WmmaM + j *
              WmmaN;
            wmma::store_matrix_sync(tile_ptr, c[i][j], ShmemStride, C_LAYOUT);
          }
        }
      }

      __syncthreads();

      // Step 4. Store the D from shared memory to global memory.
      // Now that shared memory contains all the D tiles, stream them to global 
      // memory.
      half *dst_gmem_warp_stream_ptr = &C[gmem_idx];
      half *dst_shmem_warp_ptr = (BlockRowTiles >= 16) ? (half*)&shm[0] + 
        readWarpX * WmmaM * ShmemStride + readWarpY * (WmmaN * 16): 
        (half*)&shm[0] + warpId * ShmemStride * WmmaN;

      if(BlockRowTiles >= 16){
        if(readWarpX < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy4_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){  // BlockRowTiles=8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaM; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId); 
          }
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK && laneId < c_read_lanes; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId); 
          }
        }
      }

      __syncthreads();
    } // 
#endif

  }
#endif
}



// NT. 
// 1. one warp do half on the k-dim
// 2. adjacent warps do the same dim.
template<int WmmaM,  int WmmaN, int WmmaK, int 
  WarpColTiles, int WarpRowTiles, int ChunkCol, int ChunkK>
  __global__ void __launch_bounds__(256, 1)
_gemm_wmma_shm_splitk_persistent(
    int BlockRowWarps, int BlockColWarps, 
    int WarpsPerBlock,
    int M, int N, int K,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{
  int SKEW_HALF = 8;
  const int half_col = ChunkCol>>1;

  int  m_tiles = M / WmmaM;
  int  n_tiles = N / WmmaN;

  int BlockRowTiles = (BlockRowWarps * WarpRowTiles)/2; 
  int BlockColTiles = BlockColWarps * WarpColTiles; 

  // Info of C/D.
  int ShmemStride = WmmaN * BlockRowTiles;   // C_LINE_LEN of block.
  int ShmemOffset = WmmaN * WarpRowTiles;    // C_LINE_LEN of warp.
  int c_read_lanes = BlockRowTiles * WmmaN / 4;//sizeof(half) / sizeof(int2); 

  // Info of ChunkK- A/B.
  int ChunkCopyLinesPerWarp = 512 / (ChunkK * WmmaK * 2);  
  int ChunkCopyLineLanes = (ChunkK * WmmaK * 2) / 16; // sizeof(int4);   

  // Used for A/B.
  int ShmemChunkLine = ChunkK * WmmaK + SKEW_HALF;
  extern __shared__ half shm[];

  typedef int4 copy4_t;
  typedef int2 copy2_t;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may 
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int grid1D = gridDim.x * gridDim.y;

  int tId = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpId = tId / WARP_SIZE;
  unsigned int laneId = tId & WARP_SIZE - 1;
  unsigned int warpX = warpId / BlockRowWarps;
  unsigned int warpY = (warpId / 2) & BlockRowWarps - 1; // % BlockRowWarps.

  unsigned int block_tile_i, block_tile_j, block_pos;

  // Step 1. Init: copy B to registers.
  bool loadB = false;

  wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, 
    wmma::col_major> b[WarpRowTiles][half_col];
#if 1
  for(block_pos = blockId; ; block_pos += grid1D) {
    block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) * (BlockColTiles);
    block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

    if (block_tile_i >= m_tiles) 
      break;

    half *srcBPtr = &B[block_tile_j * WmmaN * ldb];
    half *srcWarpPtr;
    size_t shmemIdx;  

#pragma unroll
    for (int tile_k = 0; tile_k < half_col; tile_k += ChunkK) {
#pragma unroll
      int nthTile = warpId / 2;
      int nth =0;
      {
        // Step 1.1: Copy B from gmem to shmem.
        // Begin shmem_idx of warp.
        shmemIdx =  warpId * WmmaN; 
        srcWarpPtr = warpId % 2 == 0 ? 
          (srcBPtr + nthTile * WmmaN * ldb + tile_k * WmmaK):
          (srcBPtr + nthTile * WmmaN * ldb + (half_col + tile_k) * WmmaK);

        // Pointer to the tile-pos of the lane.
        int4* lanePtr = (int4*)(srcWarpPtr + (laneId / ChunkCopyLineLanes) * ldb) 
          + laneId % ChunkCopyLineLanes;
        // Begin shemm_idx of lane.
        shmemIdx += laneId / ChunkCopyLineLanes;

#pragma unroll 
        for(int i = 0; i < (WmmaN / ChunkCopyLinesPerWarp); i++){
          *((int4*)(&shm[shmemIdx * ShmemChunkLine]) + laneId % 
              ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

          // Update global pointer and shmem pointer.
          lanePtr = (int4*)((half*)lanePtr + ldb * ChunkCopyLinesPerWarp);
          shmemIdx += ChunkCopyLinesPerWarp;
        }
        __syncthreads();

        // init.
        shmemIdx = warpId * WmmaN;  
#pragma unroll
        for(int k_step = 0; k_step < ChunkK; k_step++){

          const half *tilePtr = &shm[shmemIdx * ShmemChunkLine + 
            k_step * WmmaK];
          wmma::load_matrix_sync(b[nth][tile_k + k_step], tilePtr, WmmaK * 
              ChunkK + SKEW_HALF);
        }
        __syncthreads();
      }
    }
  }
#endif


  // Do load(A, C) and compute(C = A * B).
  // for: blk_tiles(blkTilesX, blkTilesY) warp_tiles(warpTilesX, warpTilesY)
  // assume: warpTilesX == blkTilesX
  //           |     Y     |              Y                       Y
  //  X=0,1,...|           | 
  // ----------|-----------|-----------------------|---------------------|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|

#if 1
  int runtimes = 1; //102720;
  for(int run = 0; run < runtimes; run++){
#pragma unroll
    for(block_pos = blockId; ; block_pos += grid1D) {
      block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) 
        * (BlockColTiles);
      block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

      // Stop when there are no more D matrix tiles to compute in this CTA.
      if (block_tile_i >= m_tiles) 
        break;

      // Step 1. Load C.
      // BlkTiles(x, y): (8, 4), (8, 8), 1<=x<=8, y = 4, 8, ...
      //               ----> readWarpY 
      // BlkRowTiles = |0     15|16    31|
      //               |--------|--------|
      //     readWarpX |  warp0 | warp1  |
      //               |  warp2 | warp3  |
      //               |  warp4 | warp5  |
      //               |  ...   |        |
      //               |  warp6 | warp7  |
      //               |--------|--------|

      // This warp's pointer  access the C and D shm tile.
      //half *shmem_warp_tile_ptr = (half*)&shm[0] + warpX * 
      //  WarpColTiles * WmmaN * ShmemStride  + warpY * ShmemOffset;

      // Copy C from gmem to shmem.
      // This idx point to the tile the warp read.
      // 16 * WMMN * 2 = 512 = 32 * 16
      int warpsReadRow = (BlockRowTiles % 16 == 0) ? 
        (BlockRowTiles / 16):
        (BlockRowTiles / 16) + 1;
      int readWarpX = warpId / warpsReadRow;
      int readWarpY = warpId % warpsReadRow;


      // This warp's pointer to the shared memory.
      half *shmem_warp_stream_ptr = (BlockRowTiles >= 16) ? (half*)&shm[0] + 
        readWarpX * WmmaM * ShmemStride + readWarpY * (WmmaN * 16): 
        (half*)&shm[0] + warpId * ShmemStride * WmmaN;

      // This warp's pointer to the C matrix that is copied to shared memory.
      size_t gmem_idx = (BlockRowTiles >= 16) ? 
        ((block_tile_i + readWarpX) * WmmaM * ldc + (block_tile_j +
          readWarpY * 16) * WmmaN) :
        (block_tile_i + warpId) * WmmaM * ldc + block_tile_j * WmmaN;
      const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, float> 
        acc[WarpColTiles][WarpRowTiles];
      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, half> 
        c[WarpColTiles][WarpRowTiles];

#if 0
      // Step 1.1  Read C from global  mem to shared mem.
      // Stream multiple C tiles to shared memory.
      if(BlockRowTiles >= 16) {
        if(readWarpX < BlockColTiles){
          //#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy4_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
            *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles && laneId < c_read_lanes){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }
      __syncthreads();

      // Step 1.2. Copy C/D from the shm to the fragment.
#if 1
#pragma unroll
      for(int i = 0;  i < WarpColTiles; i++){
#pragma unroll
        for(int j = 0; j < WarpRowTiles; j++){
          wmma::fill_fragment(acc[i][j], 0.0f);
        }
      }
#endif

      // Load the C matrix tiles into fragments from shared memory.
      // Part of warps do this.
      //if(warpX < BlockColWarps)
      {
        half *tile_ptr = shmem_warp_tile_ptr - ShmemStride  * WmmaK;
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
          tile_ptr += ShmemStride * WmmaK; 
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            wmma::load_matrix_sync(c[i][j], tile_ptr, ShmemStride, C_LAYOUT);
            tile_ptr += WmmaN;
          }
        }

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++) {
              acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
            }
          }
        }
      }
      __syncthreads();
#endif

#if 1
      for(int i = 0;  i < WarpColTiles; i++)
        for(int j = 0; j < WarpRowTiles; j++)
          wmma::fill_fragment(acc[i][j], 0.0f);
      //__syncthreads();
#endif


      // Step 2. Read A from global mem to shared mem.
      // Assume: ChunkK <= 16, so one warp can read one row_line of A/B.
      // A[BlockColTiles][ChunkK]
      // warp0(A[0][0])        ...    warp0(A[0][half_col])
      // warp1(A[0][half_col]) ...    warp1(A[0][ChunkCol])
      // warp2(A[1][0])        ...    warp2(A[1][half_col])
      // warp3(A[1][half_col]) ...    warp3(A[1][ChunkCol])

      // Pointer to the tile of A/B.
      half *srcAPtr = &A[block_tile_i * WmmaM * lda];
      half *srcTilePtr;

#pragma unroll
      for (int tile_k = 0; tile_k < half_col; tile_k += ChunkK){
        // Step 1. Copy slices of the A matrices to shared memory.
        int nthTile = warpId / 2;         
        {
          // Pointer to the tile-pos of the warp.
          srcTilePtr = (warpId % 2) == 0 ? 
            srcAPtr + (nthTile * WmmaM) * lda + tile_k * WmmaK : 
            srcAPtr + (nthTile * WmmaM) * lda + (tile_k + half_col) * WmmaK;


          // Begin shmem_idx of warp.
          size_t shmemIdx = warpId * WmmaM;
          // shemm_idx, pointer to the src_a of the lane.
          shmemIdx += laneId / ChunkCopyLineLanes;
          int4* lanePtr =  (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) *
              lda) + laneId % ChunkCopyLineLanes;

          // Do copy A/B to shmem.
#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;
            // Update global pointer and shmem pointer.
            lanePtr = (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }
        }
        __syncthreads();



        // Step 2. Compute a grid of C matrix tiles in each warp.
        wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
          wmma::row_major> a[WarpColTiles];
#pragma unroll
        for (int k_step = 0; k_step < ChunkK; k_step++) {
#pragma unroll
          for (int i = 0; i < WarpColTiles; i++) {
            // Load A from shmem to fragment.
            size_t shmem_idx_a = warpId % 2 == 0 ? i * 2 * WmmaM : (i * 2 + 1) * WmmaM;
            const half *tilePtr =  &shm[shmem_idx_a * ShmemChunkLine + k_step 
              * WmmaK];
            wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);
#pragma unroll
            for (int j = 0; j < WarpRowTiles; j++) {
              wmma::mma_sync(acc[i][j], a[i], b[j][tile_k + k_step], acc[i][j]);
            }
          }
        }
        __syncthreads();
      }


      // Step 3. Store the D fragments to shared memory.
      half *shmem_warp_tile_ptr = (half*)&shm[0] + warpX * 
        WarpColTiles * WmmaN * ShmemStride  + warpY * ShmemOffset;
      //if(warpX < BlockColWarps)
      {
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++){
              c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);
            }

            half *tile_ptr = warpId % 2 == 0 ? shmem_warp_tile_ptr + i * 2 * WmmaM *
              ShmemStride + j * WmmaN : shmem_warp_tile_ptr + (i * 2 + 1) * WmmaM *
              ShmemStride + j * WmmaN;
            wmma::store_matrix_sync(tile_ptr, c[i][j], ShmemStride, C_LAYOUT);
          }
        }
      }

      __syncthreads();

      // Step 4. Store the D from shared memory to global memory.
      // Now that shared memory contains all the D tiles, stream them to global 
      // memory.
      half *dst_gmem_warp_stream_ptr = &C[gmem_idx];
      half *dst_shmem_warp_ptr = (half*)&shm[0] + warpId * 2 * WmmaN  *
        ShmemStride;

      if(BlockRowTiles >= 16){
        if(readWarpX < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy4_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){  // BlockRowTiles=8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaM; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId); 
          }
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK && laneId < c_read_lanes; i++) {

            int2 c0 = *((int2 *)(dst_shmem_warp_ptr + ShmemStride * i) +
                laneId);
            int2 c1 =  *((int2 *)(dst_shmem_warp_ptr + WmmaN *
                  ShmemStride + ShmemStride * i) + laneId);
            half2 c_low = __hadd2(*((half2*)&c0.x), *((half2*)&c1.x));
            half2 c_high = __hadd2(*((half2*)&c0.y), *((half2*)&c1.y));
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              make_int2(*((int*)&c_low), *((int*)&c_high));

#if 0
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId); 
#endif
          }
        }
      }

      __syncthreads();
    } // 

  }
#endif
}


//
// Half of B stored in register, the other half read when needed.
template<int WmmaM,  int WmmaN, int WmmaK, int 
  WarpColTiles, int WarpRowTiles, int ChunkCol, int ChunkK>
  __global__ void __launch_bounds__(256, 1)
_gemm_wmma_shm_persistent_r(
    int BlockRowWarps, int BlockColWarps, 
    int WarpsPerBlock,
    int M, int N, int K,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{

  const int RK = 32;
  int SKEW_HALF = 8;

  int  m_tiles = M / WmmaM;
  int  n_tiles = N / WmmaN;

  int BlockRowTiles = BlockRowWarps * WarpRowTiles; 
  int BlockColTiles = BlockColWarps * WarpColTiles; 

  // Info of C/D.
  int ShmemStride = WmmaN * BlockRowTiles;   // C_LINE_LEN of block.
  int ShmemOffset = WmmaN * WarpRowTiles;    // C_LINE_LEN of warp.
  int c_read_lanes = BlockRowTiles * WmmaN / 4;//sizeof(half) / sizeof(int2); 

  // Info of ChunkK- A/B.
  int ChunkCopyLinesPerWarp = 512 / (ChunkK * WmmaK * 2);  
  int ChunkCopyLineLanes = (ChunkK * WmmaK * 2) / 16; // sizeof(int4);   


  // Used for A/B.
  int ShmemChunkLine = ChunkK * WmmaK + SKEW_HALF;
  extern __shared__ half shm[];

  typedef int4 copy4_t;
  typedef int2 copy2_t;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may 
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int grid1D = gridDim.x * gridDim.y;

  int tId = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpId = tId / WARP_SIZE;
  unsigned int laneId = tId & WARP_SIZE - 1;
  unsigned int warpX = warpId / BlockRowWarps;
  unsigned int warpY = warpId & BlockRowWarps - 1; // % BlockRowWarps.

  unsigned int block_tile_i, block_tile_j, block_pos;

  // Step 1. Init: copy B to registers.
  wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, 
    wmma::col_major> b[WarpRowTiles][RK];
#if 0
  for(block_pos = blockId; ; block_pos += grid1D) {
    block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) * (BlockColTiles);
    block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

    if (block_tile_i >= m_tiles) 
      break;

    half *srcBPtr = &B[block_tile_j * WmmaN * ldb];
    half *srcWarpPtr;
    size_t shmemIdx;  

#pragma unroll
    for (int tile_k = 0; tile_k < RK; tile_k += ChunkK) {
#pragma unroll
      for(int nthTile = warpId * WarpRowTiles, nth =0; nth < WarpRowTiles; 
          nth++, nthTile++){

        // Step 1.1: Copy B from gmem to shmem.
        // Begin shmem_idx of warp.
        shmemIdx =  warpId * WmmaN; 
        srcWarpPtr = srcBPtr + nthTile * WmmaN * ldb + tile_k * WmmaK;

        // Pointer to the tile-pos of the warp.
        int4* lanePtr = (int4*)(srcWarpPtr + (laneId / ChunkCopyLineLanes) * ldb) 
          + laneId % ChunkCopyLineLanes;
        // Begin shemm_idx of lane.
        shmemIdx += laneId / ChunkCopyLineLanes;
#if 1
#pragma unroll 
        for(int i = 0; i < (WmmaN / ChunkCopyLinesPerWarp); i++){
          *((int4*)(&shm[shmemIdx * ShmemChunkLine]) + laneId % 
              ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

          // Update global pointer and shmem pointer.
          lanePtr = (int4*)((half*)lanePtr + ldb * ChunkCopyLinesPerWarp);
          shmemIdx += ChunkCopyLinesPerWarp;
        }
#endif
        __syncthreads();

        shmemIdx = warpId * WmmaN;  // init.
#pragma unroll
        for(int k_step = 0; k_step < ChunkK; k_step++){

          const half *tilePtr = &shm[shmemIdx * ShmemChunkLine + 
            k_step * WmmaK];
          wmma::load_matrix_sync(b[nth][tile_k + k_step], tilePtr, WmmaK * 
              ChunkK + SKEW_HALF);
        }
        __syncthreads();

      }
    }
  }
#endif


  // Do load(A, C) and compute(C = A * B).
  // for: blk_tiles(blkTilesX, blkTilesY) warp_tiles(warpTilesX, warpTilesY)
  // assume: warpTilesX == blkTilesX
  //           |     Y     |              Y                       Y
  //  X=0,1,...|           | 
  // ----------|-----------|-----------------------|---------------------|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|

#if 1
  int runtimes = 1; //102720;
  for(int run = 0; run < runtimes; run++){
#pragma unroll
    for(block_pos = blockId; ; block_pos += grid1D) {
      block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) 
        * (BlockColTiles);
      block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

      // Stop when there are no more D matrix tiles to compute in this CTA.
      if (block_tile_i >= m_tiles) 
        break;

      // Step 1. Load C.
      // BlkTiles(x, y): (8, 4), (8, 8), 1<=x<=8, y = 4, 8, ...
      //               ----> readWarpY 
      // BlkRowTiles = |0     15|16    31|
      //               |--------|--------|
      //     readWarpX |  warp0 | warp1  |
      //               |  warp2 | warp3  |
      //               |  warp4 | warp5  |
      //               |  ...   |        |
      //               |  warp6 | warp7  |
      //               |--------|--------|

      // This warp's pointer  access the C and D shm tile.
      half *shmem_warp_tile_ptr = (half*)&shm[0] + warpX * 
        WarpColTiles * WmmaN * ShmemStride  + warpY * ShmemOffset;

      // Copy C from gmem to shmem.
      // This idx point to the tile the warp read.
      // 16 * WMMN * 2 = 512 = 32 * 16
      int warpsReadRow = (BlockRowTiles % 16 == 0) ? 
        (BlockRowTiles / 16):
        (BlockRowTiles / 16) + 1;
      int readWarpX = warpId / warpsReadRow;
      int readWarpY = warpId % warpsReadRow;


      // This warp's pointer to the shared memory.
      half *shmem_warp_stream_ptr = (BlockRowTiles >= 16) ? (half*)&shm[0] + 
        readWarpX * WmmaM * ShmemStride + readWarpY * (WmmaN * 16): 
        (half*)&shm[0] + warpId * ShmemStride * WmmaN;

      // This warp's pointer to the C matrix that is copied to shared memory.
      size_t gmem_idx = (BlockRowTiles >= 16) ? 
        ((block_tile_i + readWarpX) * WmmaM * ldc + (block_tile_j +
          readWarpY * 16) * WmmaN) :
        (block_tile_i + warpId) * WmmaM * ldc + block_tile_j * WmmaN;
      const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, float> 
        acc[WarpColTiles][WarpRowTiles];
      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, half> 
        c[WarpColTiles][WarpRowTiles];

#if 0
      // Step 1.1  Read C from global  mem to shared mem.
      // Stream multiple C tiles to shared memory.
      if(BlockRowTiles >= 16) {
        if(readWarpX < BlockColTiles){
          //#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy4_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
            *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles && laneId < c_read_lanes){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }
      __syncthreads();

      // Step 1.2. Copy C/D from the shm to the fragment.
#if 1
#pragma unroll
      for(int i = 0;  i < WarpColTiles; i++){
#pragma unroll
        for(int j = 0; j < WarpRowTiles; j++){
          wmma::fill_fragment(acc[i][j], 0.0f);
        }
      }
#endif

      // Load the C matrix tiles into fragments from shared memory.
      // Part of warps do this.
      //if(warpX < BlockColWarps)
      {
        half *tile_ptr = shmem_warp_tile_ptr - ShmemStride  * WmmaK;
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
          tile_ptr += ShmemStride * WmmaK; 
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            wmma::load_matrix_sync(c[i][j], tile_ptr, ShmemStride, C_LAYOUT);
            tile_ptr += WmmaN;
          }
        }

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++) {
              acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
            }
          }
        }
      }
      __syncthreads();
#endif

#if 1
      for(int i = 0;  i < WarpColTiles; i++)
        for(int j = 0; j < WarpRowTiles; j++)
          wmma::fill_fragment(acc[i][j], 0.0f);
      //__syncthreads();
#endif


      // Step 2. Read A from global mem to shared mem.
      // Assume: ChunkK <= 16, so one warp can read one row_line of A/B.
      //   A[BlockColTiles][ChunkK], B[BlockRowTiles][ChunkK] 
      //   warp0->A[0],   warp1->A[1], ...,   warpx->A[x], ...,   warpy->B[y]
      //   warp0->A[0+8], warp1->A[1+8], ..., warpx->A[x+8], ..., warpy->B[y+8]

      // Pointer to the tile of A/B.
      half *srcAPtr = &A[block_tile_i * WmmaM * lda];
      half *srcTilePtr;

#if 1
      // The first part of gemm_r.
#pragma unroll
      for (int tile_k = 0; tile_k < RK; tile_k += ChunkK) {
        // Step 1. Copy slices of the A matrices to shared memory.
#pragma unroll
        for(int nthTile = warpId; nthTile < BlockColTiles; nthTile +=
            WarpsPerBlock){

          // Pointer to the tile-pos of the warp.
          srcTilePtr = srcAPtr + (nthTile * WmmaM) * lda + tile_k * WmmaK;

          // Begin shmem_idx of warp.
          size_t shmemIdx = WmmaM * nthTile;

          // shemm_idx of each lane.
          shmemIdx += laneId / ChunkCopyLineLanes;

          // Do copy A/B to shmem.
          int4* lanePtr =  (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) *
              lda) + laneId % ChunkCopyLineLanes;
#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }
        }

        __syncthreads();

        // Step 2. Compute a grid of C matrix tiles in each warp.
        //if(warpX < BlockColWarps)
        {
          wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
            wmma::row_major> a[WarpColTiles];
#pragma unroll
          for (int k_step = 0; k_step < ChunkK; k_step++) {
#pragma unroll
            for (int i = 0; i < WarpColTiles; i++) {
              // Load A from shmem to fragment.
              size_t shmem_idx_a = warpX * WarpColTiles * WmmaM + (i * WmmaM);
              const half *tilePtr =  &shm[shmem_idx_a * ShmemChunkLine + k_step 
                * WmmaK];
              wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);
#pragma unroll
              for (int j = 0; j < WarpRowTiles; j++) {
                wmma::mma_sync(acc[i][j], a[i], b[j][tile_k + k_step], acc[i][j]);
              }
            }
          }
        }

        __syncthreads();
      }


      // The second part of gemm_r.
      half *srcBPtr = &B[block_tile_j * WmmaN * ldb];
      int totalTiles = BlockColTiles + BlockRowTiles;
      // Offset in shared memory from which the B matrix is stored.
      size_t shmem_idx_b_off = BlockColTiles * WmmaM;

#if 1
#pragma unroll
      for (int tile_k = RK; tile_k < ChunkCol; tile_k += ChunkK) {
        // Step 1. Copy slices of the A matrices to shared memory.
#pragma unroll
        for(int nthTile = warpId; nthTile < totalTiles; nthTile +=
            WarpsPerBlock){

          // Pointer to the tile-pos of the warp.
          srcTilePtr = (nthTile < BlockColTiles) ? 
            srcAPtr + (nthTile * WmmaM) * lda + tile_k * WmmaK:
            srcBPtr + (nthTile - BlockColTiles) * WmmaN * ldb + tile_k * 
            WmmaK;

          // Begin shmem_idx of warp.
          size_t shmemIdx = (nthTile < BlockColTiles)?
            (WmmaM * nthTile):
            ((WmmaM * BlockColTiles) + WmmaN * (nthTile - BlockColTiles));
          // shemm_idx of each lane.
          shmemIdx += laneId / ChunkCopyLineLanes;

          // Do copy A/B to shmem.
          int4* lanePtr =  (nthTile < BlockColTiles)?
            (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) *
                lda) + laneId % ChunkCopyLineLanes:
            (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) * ldb) +
            laneId % ChunkCopyLineLanes;
#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (nthTile < BlockColTiles) ?
              (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp):
              (int4*)((half*)lanePtr + ldb * ChunkCopyLinesPerWarp);

            shmemIdx += ChunkCopyLinesPerWarp;
          }
        }

        __syncthreads();

        // Step 2. Compute a grid of C matrix tiles in each warp.
        //if(warpX < BlockColWarps)
        {
          wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
            wmma::row_major> a[WarpColTiles];
          wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, 
            wmma::col_major> b2[WarpRowTiles];
#pragma unroll
          for (int k_step = 0; k_step < ChunkK; k_step++) {
#pragma unroll
            for (int i = 0; i < WarpColTiles; i++) {
              // Load A from shmem to fragment.
              size_t shmem_idx_a = warpX * WarpColTiles * WmmaM + (i * WmmaM);
              const half *tilePtr =  &shm[shmem_idx_a * ShmemChunkLine + k_step 
                * WmmaK];
              wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);
#pragma unroll
              for (int j = 0; j < WarpRowTiles; j++) {
                if(i == 0){
                  size_t shmem_idx_b = shmem_idx_b_off + warpY * (WarpRowTiles *
                      WmmaN) + j * WmmaN; 
                  const half *tilePtr = &shm[shmem_idx_b * ShmemChunkLine + 
                    k_step * WmmaK];
                  wmma::load_matrix_sync(b2[j], tilePtr, WmmaK * ChunkK + 
                      SKEW_HALF);
                }
                wmma::mma_sync(acc[i][j], a[i], b2[j], acc[i][j]);
              }
            }
          }
        }

        __syncthreads();
      }
#endif


      // Step 3. Store the D fragments to shared memory.
      //if(warpX < BlockColWarps)
      {
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            // Uniform, point-wise transformations of ALL fragment elements by ALL 
            // threads in the warp are well-defined even though element indices 
            // within fragment storage are not defined.
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++){
              c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);
            }

            half *tile_ptr = shmem_warp_tile_ptr + i * ShmemStride * WmmaM + j *
              WmmaN;
            wmma::store_matrix_sync(tile_ptr, c[i][j], ShmemStride, C_LAYOUT);
          }
        }
      }

      __syncthreads();

      // Step 4. Store the D from shared memory to global memory.
      // Now that shared memory contains all the D tiles, stream them to global 
      // memory.
      half *dst_gmem_warp_stream_ptr = &C[gmem_idx];
      half *dst_shmem_warp_ptr = (BlockRowTiles >= 16) ? (half*)&shm[0] + 
        readWarpX * WmmaM * ShmemStride + readWarpY * (WmmaN * 16): 
        (half*)&shm[0] + warpId * ShmemStride * WmmaN;

      if(BlockRowTiles >= 16){
        if(readWarpX < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy4_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){  // BlockRowTiles=8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaM; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId); 
          }
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles)
        {
#pragma unroll
          for (int i = 0; i < WmmaK && laneId < c_read_lanes; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(dst_shmem_warp_ptr + ShmemStride * i) + laneId); 
          }
        }
      }


      __syncthreads();
    } // 
#endif

  }
#endif
}


// NT. 
// Opt: double buffer.
template<int WmmaM,  int WmmaN, int WmmaK, int 
  WarpColTiles, int WarpRowTiles, int ChunkCol, int ChunkK>
  __global__ void __launch_bounds__(256, 1)
_gemm_wmma_shm_persistent_db(
    int BlockRowWarps, int BlockColWarps, 
    int WarpsPerBlock,
    int M, int N, int K,
    float alpha,
    half *A, int lda, 
    half *B, int ldb, 
    float beta,
    half *C, int ldc)
{
  const int SKEW_HALF = 8;
  int  m_tiles = M / WmmaM;
  int  n_tiles = N / WmmaN;

  int BlockRowTiles = BlockRowWarps * WarpRowTiles; 
  int BlockColTiles = BlockColWarps * WarpColTiles; 


  extern __shared__ half shm[];

  typedef int4 copy4_t;
  typedef int2 copy2_t;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may 
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the 
  // matrix to the right and down, and selects the next tile to compute. Once 
  // there's no such tile, all warps in this CTA exit.
  int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  int grid1D = gridDim.x * gridDim.y;

  int tId = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpId = tId / WARP_SIZE;
  unsigned int laneId = tId & WARP_SIZE - 1;
  unsigned int warpX = warpId / BlockRowWarps;
  unsigned int warpY = warpId & BlockRowWarps - 1; // % BlockRowWarps.

  unsigned int block_tile_i, block_tile_j, block_pos;

  // Step 1. Init: copy B to registers.
  //bool loadB = false;

  wmma::fragment<wmma::matrix_b, WmmaM, WmmaN, WmmaK, half, 
    wmma::col_major> b[WarpRowTiles][ChunkCol];
#if 1
  for(block_pos = blockId; ; block_pos += grid1D) 
  {
    block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) * (BlockColTiles);
    block_tile_j = (block_pos * BlockRowTiles) & n_tiles - 1;

    if (block_tile_i >= m_tiles) 
      break;

    half *srcBPtr = &B[block_tile_j * WmmaN * ldb];
    half *srcWarpPtr;
    size_t shmemIdx;  
    const  int chunkB = 4;
    const int ChunkBLineLanes = 8; // WmmaK * 2 * chunkB / sizeof(int4).
    const int ChunkBLinesPerWarp = 4; // 32 / ChunkBLineLanes
    const int ShmemChunkBLine = chunkB * WmmaK + SKEW_HALF;
    int laneChunkLineId = laneId & ChunkBLineLanes - 1;

#pragma unroll
    for(int nthTile = warpId * WarpRowTiles, nth =0; nth < WarpRowTiles; 
        nth++, nthTile++){

      // += chunkB * 2
#pragma unroll
      for (int tile_k = 0; tile_k < ChunkCol; tile_k += 8)  {

        // Step 1.1: Copy B from gmem to shmem.
        // Begin shmem_idx of warp.
        shmemIdx =  warpId * WmmaN; 
        srcWarpPtr = srcBPtr + nthTile * WmmaN * ldb + tile_k * WmmaK;

        // Pointer to the tile-pos of the warp.
        //int4* lanePtr = (int4*)(srcWarpPtr + (laneId / ChunkBLineLanes) * ldb) 
        int4* lanePtr = (int4*)(srcWarpPtr + (laneId>>3) * ldb) +
          laneChunkLineId;
        // Begin shemm_idx of lane.
        //shmemIdx += laneId / ChunkBLineLanes;
        shmemIdx += (laneId>>3);
#if 1
#pragma unroll 
        for(int i = 0; i < (WmmaN>>2); i++){
          //*((int4*)(&shm[shmemIdx * ShmemChunkBLine]) + laneId % 
          //    ChunkBLineLanes) = __ldg(lanePtr); //*lanePtr;
          *((int4*)(&shm[shmemIdx * ShmemChunkBLine]) + laneChunkLineId) = 
            __ldg(lanePtr); //*lanePtr;

          // Update global pointer and shmem pointer.
          lanePtr = (int4*)((half*)lanePtr + ldb * ChunkBLinesPerWarp);
          shmemIdx += ChunkBLinesPerWarp;
        }
#endif
        //__syncthreads();
        __syncwarp();

        shmemIdx = warpId * WmmaN;  // init.
#pragma unroll(4)
        for(int k_step = 0; k_step < chunkB; k_step++){

          const half *tilePtr = &shm[shmemIdx * ShmemChunkBLine + 
            k_step * WmmaK];
          wmma::load_matrix_sync(b[nth][tile_k + k_step], tilePtr, WmmaK * 
              chunkB + SKEW_HALF);
        }
        //__syncthreads();

        // 2nd shm buffer.
        shmemIdx =  warpId * WmmaN + (WmmaN<<3); 
        srcWarpPtr = srcBPtr + nthTile * WmmaN * ldb + (tile_k + 4) * WmmaK;
        // Pointer to the tile-pos of the warp.
        lanePtr = (int4*)(srcWarpPtr + (laneId / ChunkBLineLanes) * ldb) 
          + laneChunkLineId;
        // Begin shemm_idx of lane.
        shmemIdx += laneId / ChunkBLineLanes;

#if 1
#pragma unroll 
        for(int i = 0; i < (WmmaN>>2); i++){
          *((int4*)(&shm[shmemIdx * ShmemChunkBLine]) + laneChunkLineId) = 
            __ldg(lanePtr); //*lanePtr;

          // Update global pointer and shmem pointer.
          lanePtr = (int4*)((half*)lanePtr + ldb * ChunkBLinesPerWarp);
          shmemIdx += ChunkBLinesPerWarp;
        }
        //__syncthreads();
        __syncwarp();
#endif

        shmemIdx = warpId * WmmaN;  // init.
#pragma unroll
        for(int k_step = 0; k_step < chunkB; k_step++){

          const half *tilePtr = &shm[shmemIdx * ShmemChunkBLine + 
            k_step * WmmaK];
          wmma::load_matrix_sync(b[nth][tile_k + k_step], tilePtr, WmmaK * 
              chunkB + SKEW_HALF);
        }

        //__syncthreads();
      } // chunkB
    }

  }
#endif
  for(int i = 0; i < WarpRowTiles; i++)
    for(int j = 0; j < ChunkCol; j++){
      wmma::fill_fragment(b[i][j], 0.0f);
    }
  __syncthreads();


  // Do load(A, C) and compute(C = A * B).
  // for: blk_tiles(blkTilesX, blkTilesY) warp_tiles(warpTilesX, warpTilesY)
  // assume: warpTilesX == blkTilesX
  //           |     Y     |              Y                       Y
  //  X=0,1,...|           | 
  // ----------|-----------|-----------------------|---------------------|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|
  // warpx     |warp0 warp1|warp0 warp1 warp2 warp3|warp0 warp1 ... warp7|

#if 1
  int runtimes = 1; //102720;
  for(int run = 0; run < runtimes; run++)
  {
#pragma unroll
    for(block_pos = blockId; ; block_pos += grid1D) {
      block_tile_i = ((block_pos * BlockRowTiles) / n_tiles) 
        * (BlockColTiles);
      block_tile_j = (block_pos * BlockRowTiles) % n_tiles;

      // Stop when there are no more D matrix tiles to compute in this CTA.
      if (block_tile_i >= m_tiles) 
        break;

      // Info of C/D, ChunkK(A)
      int ShmemStride = WmmaN * BlockRowTiles;   // C_LINE_LEN of block.
      int ShmemOffset = WmmaN * WarpRowTiles;    // C_LINE_LEN of warp.
      int c_read_lanes = BlockRowTiles * WmmaN / 4;//sizeof(half) / sizeof(int2); 

      // Info of ChunkK- A.
      int ChunkCopyLinesPerWarp = 512 / (ChunkK * WmmaK * 2);  
      int ChunkCopyLineLanes = (ChunkK * WmmaK * 2) / 16; // sizeof(int4);   
      int ShmemChunkLine = ChunkK * WmmaK + SKEW_HALF;


      // Step 1. Load C.
      // BlkTiles(x, y): (8, 4), (8, 8), 1<=x<=8, y = 4, 8, ...
      //               ----> readWarpY 
      // BlkRowTiles = |0     15|16    31|
      //               |--------|--------|
      //     readWarpX |  warp0 | warp1  |
      //               |  warp2 | warp3  |
      //               |  warp4 | warp5  |
      //               |  ...   |        |
      //               |  warp6 | warp7  |
      //               |--------|--------|

      // This warp's pointer  access the C and D shm tile.
      half *shmem_warp_tile_ptr = (half*)&shm[0] + warpX * 
        WarpColTiles * WmmaN * ShmemStride  + warpY * ShmemOffset;

      // Copy C from gmem to shmem.
      // This idx point to the tile the warp read.
      // 16 * WMMN * 2 = 512 = 32 * 16
      int warpsReadRow = (BlockRowTiles % 16 == 0) ? 
        (BlockRowTiles / 16):
        (BlockRowTiles / 16) + 1;
      int readWarpX = warpId / warpsReadRow;
      int readWarpY = warpId % warpsReadRow;


      // This warp's pointer to the shared memory.
      half *shmem_warp_stream_ptr = (BlockRowTiles >= 16) ? (half*)&shm[0] + 
        readWarpX * WmmaM * ShmemStride + readWarpY * (WmmaN * 16): 
        (half*)&shm[0] + warpId * ShmemStride * WmmaN;

      // This warp's pointer to the C matrix that is copied to shared memory.
      size_t gmem_idx = (BlockRowTiles >= 16) ? 
        ((block_tile_i + readWarpX) * WmmaM * ldc + (block_tile_j +
          readWarpY * 16) * WmmaN) :
        (block_tile_i + warpId) * WmmaM * ldc + block_tile_j * WmmaN;
      const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, float> 
        acc[WarpColTiles][WarpRowTiles];
      wmma::fragment<wmma::accumulator, WmmaM, WmmaN, WmmaK, half> 
        c[WarpColTiles][WarpRowTiles];

#if 0
      // Step 1.1  Read C from global  mem to shared mem.
      // Stream multiple C tiles to shared memory.
      if(BlockRowTiles >= 16) {
        if(readWarpX < BlockColTiles){
          //#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy4_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){
#pragma unroll
        for (int i = 0; i < WmmaK; i++) {
          *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
            *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles && laneId < c_read_lanes){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId) = 
              *((copy2_t *)(src_gmem_warp_stream_ptr + ldc * i) + laneId);
          }
        }
      }
      __syncthreads();

      // Step 1.2. Copy C/D from the shm to the fragment.
      // Load the C matrix tiles into fragments from shared memory.
      // Part of warps do this.
      //if(warpX < BlockColWarps)
      {
        half *tile_ptr = shmem_warp_tile_ptr - ShmemStride  * WmmaK;
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
          tile_ptr += ShmemStride * WmmaK; 
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            wmma::load_matrix_sync(c[i][j], tile_ptr, ShmemStride, C_LAYOUT);

            // Scale the C matrix.
            for (int t = 0; t < c[i][j].num_elements; t++) {
              acc[i][j].x[t] = __half2float(c[i][j].x[t]) * beta;
            }
            tile_ptr += WmmaN;
          }
        }
      }
#endif
      __syncthreads();

      // Better performance when turn on.
#if 1
      for(int i = 0;  i < WarpColTiles; i++)
        for(int j = 0; j < WarpRowTiles; j++)
          wmma::fill_fragment(acc[i][j], 0.0f);
      __syncthreads();
#endif

      // Step 2. Read A from global mem to shared mem.
      // Assume: ChunkK <= 16, so one warp can read one row_line of A/B.
      //   A[BlockColTiles][ChunkK], B[BlockRowTiles][ChunkK] 
      //   warp0->A[0],   warp1->A[1], ...,   warpx->A[x], ...,   warpy->B[y]
      //   warp0->A[0+8], warp1->A[1+8], ..., warpx->A[x+8], ..., warpy->B[y+8]

      // Pointer to the tile of A/B.
      half *srcAPtr = &A[block_tile_i * WmmaM * lda];
      half *srcBPtr = &B[block_tile_j * WmmaN * ldb];
      half *srcTilePtr;
#if 1
#pragma unroll
      for (int tile_k = 0; tile_k < ChunkCol; tile_k += (ChunkK<<1) ) {

        // First  shm buffer.
        //  Copy slices of the A matrices to shared memory.
#pragma unroll
        for(int nthTile = warpId; nthTile < BlockColTiles; nthTile +=
            WarpsPerBlock){

          // Pointer to the tile-pos of the warp.
          srcTilePtr = srcAPtr + (nthTile * WmmaM) * lda + tile_k * WmmaK;
          size_t shmemIdx = WmmaM * nthTile;       // Begin shmem_idx of warp.
          shmemIdx += laneId / ChunkCopyLineLanes; // shemm_idx of each lane.

          // Do copy A to shmem.
          int4* lanePtr =  (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) *
              lda) + laneId % ChunkCopyLineLanes;
#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }
        }
        __syncthreads();

        // Step 2. Compute a grid of C matrix tiles in each warp.
        //if(warpX < BlockColWarps)
        {
          wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
            wmma::row_major> a[WarpColTiles];
#pragma unroll
          for (int k_step = 0; k_step < ChunkK; k_step++) {
#pragma unroll
            for (int i = 0; i < WarpColTiles; i++) {
              size_t shmem_idx_a = warpX * WarpColTiles * WmmaM + (i * WmmaM);
              const half *tilePtr =  &shm[shmem_idx_a * ShmemChunkLine + k_step 
                * WmmaK];
              wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);
#pragma unroll
              for (int j = 0; j < WarpRowTiles; j++) {
                wmma::mma_sync(acc[i][j], a[i], b[j][tile_k + k_step], acc[i][j]);
              }
            }
          }
        }
        //__syncthreads();

        //  shmem-buffer-2.
        // Step 1. Copy slices of the A matrices to shared memory.
#pragma unroll
        for(int nthTile = warpId; nthTile < BlockColTiles; nthTile +=
            WarpsPerBlock){
          // Pointer to the tile-pos of the warp.
          srcTilePtr = srcAPtr + (nthTile * WmmaM) * lda + (tile_k + ChunkK) * 
            WmmaK;

          // Begin shmem_idx of warp.
          size_t shmemIdx = (nthTile + ChunkK) * WmmaM; 
          shmemIdx += laneId / ChunkCopyLineLanes; // shemm_idx of each lane.

          // Do copy A to shmem.
          int4* lanePtr =  (int4*)(srcTilePtr + (laneId / ChunkCopyLineLanes) *
              lda) + laneId % ChunkCopyLineLanes;
#pragma unroll 
          for(int i = 0; i < (WmmaM / ChunkCopyLinesPerWarp); i++){
            *((int4*)&shm[shmemIdx * ShmemChunkLine] + laneId % 
                ChunkCopyLineLanes) = __ldg(lanePtr); //*lanePtr;

            // Update global pointer and shmem pointer.
            lanePtr = (int4*)((half*)lanePtr + lda * ChunkCopyLinesPerWarp);
            shmemIdx += ChunkCopyLinesPerWarp;
          }
        }
        __syncthreads();

        // Step 2. Compute a grid of C matrix tiles in each warp.
        //if(warpX < BlockColWarps)
        {
          wmma::fragment<wmma::matrix_a, WmmaM, WmmaN, WmmaK, half, 
            wmma::row_major> a[WarpColTiles];
#pragma unroll
          for (int k_step = 0; k_step < ChunkK; k_step++) {
#pragma unroll
            for (int i = 0; i < WarpColTiles; i++) {
              size_t shmem_idx_a = ChunkK * WmmaM + warpX * WarpColTiles * WmmaM
                + (i * WmmaM);
              const half *tilePtr =  &shm[shmem_idx_a * ShmemChunkLine + k_step 
                * WmmaK];
              wmma::load_matrix_sync(a[i], tilePtr, WmmaK * ChunkK + SKEW_HALF);
#pragma unroll
              for (int j = 0; j < WarpRowTiles; j++) {
                wmma::mma_sync(acc[i][j], a[i], b[j][tile_k + k_step + ChunkK],
                    acc[i][j]);
              }
            }
          }
        }
        //__syncthreads();

      }

      // Step 3. Store the D fragments to shared memory.
      //if(warpX < BlockColWarps)
      {
#pragma unroll
        for (int i = 0; i < WarpColTiles; i++) {
#pragma unroll
          for (int j = 0; j < WarpRowTiles; j++) {
            // Uniform, point-wise transformations of ALL fragment elements by ALL 
            // threads in the warp are well-defined even though element indices 
            // within fragment storage are not defined.
#pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++){
              c[i][j].x[t] = __float2half(acc[i][j].x[t] * alpha);
            }

            half *tile_ptr = shmem_warp_tile_ptr + i * ShmemStride * WmmaM + j *
              WmmaN;
            wmma::store_matrix_sync(tile_ptr, c[i][j], ShmemStride, C_LAYOUT);
          }
        }
      }

      __syncthreads();

      // Step 4. Store the D from shared memory to global memory.
      // Now that shared memory contains all the D tiles, stream them to global 
      // memory.
      half *dst_gmem_warp_stream_ptr = &C[gmem_idx];
      if(BlockRowTiles >= 16){
        if(readWarpX < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy4_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy4_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId);
          }
        }
      }else if(BlockRowTiles == 8){  // BlockRowTiles=8
        if(warpId < BlockColTiles){
#pragma unroll
          for (int i = 0; i < WmmaK; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId); 
          }
        }
      }else{  // BlockRowTiles < 8
        if(warpId < BlockColTiles)
        {
#pragma unroll
          for (int i = 0; i < WmmaK && laneId < c_read_lanes; i++) {
            *((copy2_t *)(dst_gmem_warp_stream_ptr + ldc * i) + laneId) = 
              *((copy2_t *)(shmem_warp_stream_ptr + ShmemStride * i) + laneId); 
          }
        }
      }
      __syncthreads(); // Not need for persist.
    } // 
#endif

  }
#endif
}



// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is 
//      designed for demonstration purposes only to show the CUDA WMMA API use 
//      without relying on availability of the shared memory.
__global__ void _gemm_wmma(half *a, half *b, float *c, float *d, int m_ld, 
    int n_ld, int k_ld, float alpha, float beta)
{
  // Kernel configure.
  //int warpSize = 32;
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  // Leading dimensions. Packed with no transpositions.
  //int lda = m_ld;
  //int ldb = k_ld;
  //int ldc = n_ld;

  int lda = k_ld;
  int ldb = n_ld;
  int ldc = n_ld;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
    wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
    wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i; 

    //int bRow = warpN * WMMA_N;
    //int bCol = i;

    int bCol= warpN * WMMA_N;
    int bRow= i;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    }
  }

  // Load in the current value of c, scale it by beta, and add this our 
  // result scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, 
        wmma::mem_row_major);

    for(int i=0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, 
        wmma::mem_row_major);
  }
}



// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
//  Note: This is a less performant version of the compute_gemm kernel. It is 
//      designed for demonstration purposes only to show the CUDA WMMA API use 
//      without relying on  availability of the shared memory.
__global__ void _gemm_wmma_half(bool OP_A_N,bool OP_B_N, half *a, half *b, 
    half *c, half *d, int m_ld, int n_ld, int k_ld, float alpha, float beta, int
    lda, int ldb, int ldc)
{
  // Kernel configure.
  //int warpSize = 32;
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  // Leading dimensions. Packed with no transpositions.
  //int lda = m_ld;
  //int ldb = k_ld;
  //int ldc = n_ld;

  //int lda = k_ld;
  //int ldb = n_ld;
  //int ldc = n_ld;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
    wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, 
    wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i; 

    //int bRow = warpN * WMMA_N;
    //int bCol = i;

    int bCol = warpN * WMMA_N;
    int bRow = i;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    }
  }

  // Load in the current value of c, scale it by beta, and add this our 
  // result scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, 
        wmma::mem_row_major);

    for(int i=0; i < c_frag.num_elements; i++) {
      //c_frag.x[i] = __float2half(alpha * acc_frag.x[i] + beta * 
      //(float)(c_frag.x[i]));
      c_frag.x[i] = __float2half(alpha * acc_frag.x[i] + beta * 
          (float)(c_frag.x[i]));
      //c_frag.x[i] = __float2half(alpha * acc_frag.x[i] );
    }

    // Store the output
    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, 
        wmma::mem_row_major);
  }
}



void gemm_wmma_shm(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *a, int lda,
    half *b, int ldb,
    float beta,
    float *c, int ldc)
{
#if 0
  int SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M) *
      (CHUNK_K * K + SKEW_HALF) * 2,
      M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS *
        WARP_COL_TILES) * sizeof(float));

  checkKernelErrors((_gemm_wmma_shm<<<40, THREADS_PER_BLOCK, 
        SHMEM_SZ>>>(a, b, c, d, alpha, beta)));
#endif
}

int gemm_wmma_shm_half(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;
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


  int SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * WMMA_M) * 
      (CHUNK_K * WMMA_K + SKEW_HALF) * 2,
      WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * 
      (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(half));

  //printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

  checkKernelErrors((_gemm_wmma_shm_half<<<40, THREADS_PER_BLOCK, SHMEM_SZ>>>(
          0, 0, M, N, K, 
          alpha, A, lda, B, ldb, beta, C, ldc)));

  return 0;

}



// better than 128_16
int gemm_wmma_shm_half_config(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{

  // Default config.
  const int wmma_m = 16;
  const int wmma_n = 16;
  const int wmma_k = 16;
  //const int warp_tiles = 8;

  // Default tile size: [16, 16, 16]
  // {M, N, K} % 16 == {0, 0, 0} 
  if((M % 16) || (N % 16) || (K % 16)){
    std::cout<<"M, N, K are illegal \n";
    return 0;
  }

  int M_tiles = M / 16;
  int N_tiles = N / 16;
  int K_tiles = K / 16;

  Configure cfg;
  cfg.wmma_m_ = 16;
  cfg.wmma_n_ = 16;
  cfg.wmma_k_ = 16;
  cfg.warp_size_ = 32;

  cfg.chunk_k_ = 4;

  // blockDim.x * blockDim.y == 256
  if(M_tiles * N_tiles >= 64 && N_tiles >= 16){
    cfg.blk_warps_ = 8;

    if(M_tiles <= 8){

      if(M_tiles == 2){
        cfg.blk_row_warps_ = 4;
        cfg.blk_col_warps_ = 2;
        cfg.warp_row_tiles_ = 8;
        cfg.warp_col_tiles_ = 1;
      }else if(M_tiles == 4){
        cfg.warp_col_tiles_ = 2;
        cfg.warp_row_tiles_ = 4;
        cfg.blk_col_warps_ = 2;
        cfg.blk_row_warps_ = 4;
      }else if(M_tiles == 8){
        cfg.blk_row_warps_ = 2;
        cfg.blk_col_warps_ = 4;
        cfg.warp_col_tiles_ = 2;
        cfg.warp_row_tiles_ = 4;
      }else{
        cfg.blk_row_warps_ = 8;
        cfg.blk_col_warps_ = 1;
        cfg.warp_row_tiles_ = 8;
        cfg.warp_col_tiles_ = 1 ;
      }
    }else{
      cfg.blk_row_warps_ = 2;
      cfg.blk_col_warps_ = 4;
      cfg.warp_row_tiles_ = 4;
      cfg.warp_col_tiles_ = 2;
    }
  }
  else{
    std::cout<<"\n***** Not support now, TBD later ***** \n";
  }

  int skew_half = 8;

  int block_row_tiles = (cfg.warp_row_tiles_ * cfg.blk_row_warps_); // 8
  int block_col_tiles = (cfg.warp_col_tiles_ * cfg.blk_col_warps_); // 8

  int shmem_sz = 0;
  bool USE_SHM = true;
  shmem_sz = max(sizeof(half) * ((block_col_tiles + block_row_tiles) * 
        cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
      cfg.wmma_m_ * (cfg.blk_row_warps_ * cfg.warp_row_tiles_) *
      cfg.wmma_n_ * (cfg.blk_col_warps_ * cfg.warp_col_tiles_) *
      sizeof(half));
#if 1
  while(shmem_sz > 64 * 1024UL && USE_SHM){
    if(cfg.chunk_k_ >= 2){
      cfg.chunk_k_ /= 2;
      shmem_sz = max(sizeof(half) * ((block_col_tiles + block_row_tiles) * 
            cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
          cfg.wmma_m_ * (cfg.blk_row_warps_ * cfg.warp_row_tiles_) *
          cfg.wmma_n_ * (cfg.blk_col_warps_ * cfg.warp_col_tiles_) *
          sizeof(half));
    }
    else
      USE_SHM = false;
  }
#endif

  //shmem_sz = 64 * 1024UL; 
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
  printf("blk_row_tiles = %d, blk_col_tiles = %d \n", block_row_tiles,
      block_col_tiles);
  printf("chunk_k = %d \n", cfg.chunk_k_);
  int runtimes = 100;
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 grid(40,1);
  dim3 block(256, 1, 1);
  for(int i = 0 ; i < runtimes ; i++){
    if(USE_SHM){
      if(cfg.warp_col_tiles_ == 2 && cfg.warp_row_tiles_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_128_16<wmma_m, 
              wmma_n, wmma_k, 2, 4>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_128_16<wmma_m, wmma_n, wmma_k,  
              2, 4> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_row_warps_, cfg.blk_col_warps_,
                cfg.blk_warps_, //cfg.warp_size_,
                cfg.chunk_k_,
                // 0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_col_tiles_ == 1 && cfg.warp_row_tiles_ == 8){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_config<wmma_m, 
              wmma_n, wmma_k, 1, 8>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_config<wmma_m, wmma_n, wmma_k,  
              1, 8> 
              <<<grid, 256, shmem_sz>>>(
                cfg.blk_row_warps_, cfg.blk_col_warps_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else{
        printf("The warp_row/col_tiles is not supported within gemm \n");
      }

    }else{
      printf("Not use shm within gemm \n");
    }
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds/runtimes);
  printf("TFLOPS: %.2f\n", (((double)M * N * K *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);


  return 0;

}


// blockDim.x * blockDim.y = 128
// Tile_C[x,y]: x * y = 16;
int gemm_wmma_shm_half_128_16(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{

  // Default config.
  const int wmma_m = 16;
  const int wmma_n = 16;
  const int wmma_k = 16;

  // Default tile size: [16, 16, 16]
  if((M % 16) || (N % 16) || (K % 16)){
    std::cout<<"M, N, K are illegal \n";
    return 0;
  }

  int M_tiles = M / 16;
  int N_tiles = N / 16;

  if(M_tiles * N_tiles < 16){
    std::cout<<"\n***** Not support now, TBD later ***** \n";
    return 0;
  }

  Configure cfg;
  //cfg.wmma_m_ = 16;
  //cfg.wmma_n_ = 16;
  //cfg.wmma_k_ = 16;

  cfg.warp_size_ = 32;
  cfg.blk_warps_ = 8;
  cfg.blk_tiles_ = 64;

  // Assume
  // blockDim.x * blockDim.y == 128
  // 
  cfg.chunk_k_ = 4;

  if(M_tiles >= 4){
#if 1
    cfg.blk_tiles_x_ = 8;
    cfg.blk_tiles_y_ = 8;
    cfg.warp_tiles_x_ = 2;
    cfg.warp_tiles_y_ = 4;
#endif
  }else if(M_tiles == 2){
    cfg.blk_tiles_x_ = 2;
    cfg.blk_tiles_y_ = 8; //cfg.blk_tiles / cfg.blk_tiles_x;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 2;
  }else{
    cfg.blk_tiles_x_ = 4; 
    cfg.blk_tiles_y_ = 4;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 2 ;
  }
  cfg.blk_warps_x_ = cfg.blk_tiles_x_ / cfg.warp_tiles_x_;
  cfg.blk_warps_y_ = cfg.blk_tiles_y_ / cfg.warp_tiles_y_;

  int skew_half = 8;

  int shmem_sz = 0;
  bool USE_SHM = true;
  shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
        cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
      cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
      cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
      sizeof(half));
#if 1
  while(shmem_sz > 64 * 1024UL && USE_SHM){
    if(cfg.chunk_k_ >= 2){
      cfg.chunk_k_ /= 2;
      shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
            cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
          cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
          cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
          sizeof(half));
    }
    else
      USE_SHM = false;
  }
#endif

#if 0
  shmem_sz = 64 * 1024UL; 
  //shmem_sz = 64 * 1024UL; 
  printf("blk_row_tiles = %d, blk_col_tiles = %d \n", cfg.blk_tiles_y_,
      cfg.blk_tiles_x_);
  printf("warp_row_tiles = %d, warp_col_tiles = %d \n", cfg.warp_tiles_y_,
      cfg.warp_tiles_x_);
  printf("blk_row_warps = %d, blk_col_warps = %d \n", cfg.blk_warps_y_,
      cfg.blk_warps_x_);
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
  printf("chunk_k = %d \n", cfg.chunk_k_);
#endif
  int runtimes = 100;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 grid(24, 1);
  dim3 block(256, 1, 1);

  for(int i = 0 ; i < runtimes ; i++){

    if(USE_SHM){
      if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_128_16<wmma_m, 
              wmma_n, wmma_k, 2, 4>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_128_16<wmma_m, wmma_n, wmma_k,  
              2, 4> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }
      else if(cfg.warp_tiles_x_ == 4 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_128_16<wmma_m, 
              wmma_n, wmma_k, 4, 1>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_128_16<wmma_m, wmma_n, wmma_k,  
              4, 1> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }

      else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_128_16<wmma_m, 
              wmma_n, wmma_k, 2, 2>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_128_16<wmma_m, wmma_n, wmma_k,  
              2, 2> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                cfg.chunk_k_,
                // 0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
#if 1
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_128_16<wmma_m, 
              wmma_n, wmma_k, 1, 2>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_128_16<wmma_m, wmma_n, wmma_k,  
              1, 2> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }

      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_128_16<wmma_m, 
              wmma_n, wmma_k, 1, 1>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_128_16<wmma_m, wmma_n, wmma_k,  
              1, 1> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_128_16<wmma_m, 
              wmma_n, wmma_k, 2, 1>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_128_16<wmma_m, wmma_n, wmma_k,  
              2, 1> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }     else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_half_128_16<wmma_m, 
              wmma_n, wmma_k, 1, 4>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_half_128_16<wmma_m, wmma_n, wmma_k,  
              1, 4> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }else{
        printf("The warp_row/col_tiles is not supported within gemm \n");
      }
#endif

    }else{
      printf("Not use shm within gemm \n");
    }

  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds/runtimes);
  printf("TFLOPS: %.2f\n", (((double)M * N * K *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);



  return 0;

}


// blockDim.x * blockDim.y = 128
// Tile_C[x,y]: x * y = 16;
int gemm_wmma_shm_r_opt(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{

  // Default config.
  const int wmma_m = 16;
  const int wmma_n = 16;
  const int wmma_k = 16;

  // Default tile size: [16, 16, 16]
  if((M % 16) || (N % 16) || (K % 16)){
    std::cout<<"M, N, K are illegal \n";
    return 0;
  }

  int M_tiles = M / 16;
  int N_tiles = N / 16;

  if(M_tiles * N_tiles < 16){
    std::cout<<"\n***** Not support now, TBD later ***** \n";
    return 0;
  }

  Configure cfg;
  //cfg.wmma_m_ = 16;
  //cfg.wmma_n_ = 16;
  //cfg.wmma_k_ = 16;

  cfg.warp_size_ = 32;
  cfg.blk_warps_ = 8;
  cfg.blk_tiles_ = 64;

  // Assume
  // blockDim.x * blockDim.y == 128
  // 
  cfg.chunk_k_ = 8;

  if(M_tiles >= 4){
#if 1
    cfg.blk_tiles_x_ = 8;
    cfg.blk_tiles_y_ = 8;
    cfg.warp_tiles_x_ = 2;
    cfg.warp_tiles_y_ = 4;
#endif
  }else if(M_tiles == 2){
    cfg.blk_tiles_x_ = 2;
    cfg.blk_tiles_y_ = 8; //cfg.blk_tiles / cfg.blk_tiles_x;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 2;
  }else{
    cfg.blk_tiles_x_ = 4; 
    cfg.blk_tiles_y_ = 4;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 2 ;
  }
  cfg.blk_warps_x_ = cfg.blk_tiles_x_ / cfg.warp_tiles_x_;
  cfg.blk_warps_y_ = cfg.blk_tiles_y_ / cfg.warp_tiles_y_;

  int skew_half = 8;

  int shmem_sz = 0;
  bool USE_SHM = true;
  shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
        cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
      cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
      cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
      sizeof(half));
#if 1
  while(shmem_sz > 64 * 1024UL && USE_SHM){
    if(cfg.chunk_k_ >= 2){
      cfg.chunk_k_ /= 2;
      shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
            cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
          cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
          cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
          sizeof(half));
    }
    else
      USE_SHM = false;
  }
#endif

#if 0
  shmem_sz = 64 * 1024UL; 
  //shmem_sz = 64 * 1024UL; 
  printf("blk_row_tiles = %d, blk_col_tiles = %d \n", cfg.blk_tiles_y_,
      cfg.blk_tiles_x_);
  printf("warp_row_tiles = %d, warp_col_tiles = %d \n", cfg.warp_tiles_y_,
      cfg.warp_tiles_x_);
  printf("blk_row_warps = %d, blk_col_warps = %d \n", cfg.blk_warps_y_,
      cfg.blk_warps_x_);
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
  printf("chunk_k = %d \n", cfg.chunk_k_);
#endif
  int runtimes = 100;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 grid(24, 1);
  dim3 block(256, 1, 1);

  for(int i = 0 ; i < runtimes ; i++){

    if(USE_SHM){
      if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_r_opt<wmma_m, 
              wmma_n, wmma_k, 2, 4>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_r_opt<wmma_m, wmma_n, wmma_k,  
              2, 4> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }
      else if(cfg.warp_tiles_x_ == 4 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_r_opt<wmma_m, 
              wmma_n, wmma_k, 4, 1>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_r_opt<wmma_m, wmma_n, wmma_k,  
              4, 1> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }

      else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_r_opt<wmma_m, 
              wmma_n, wmma_k, 2, 2>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_r_opt<wmma_m, wmma_n, wmma_k,  
              2, 2> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                cfg.chunk_k_,
                // 0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
#if 1
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_r_opt<wmma_m, 
              wmma_n, wmma_k, 1, 2>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_r_opt<wmma_m, wmma_n, wmma_k,  
              1, 2> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }

      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_r_opt<wmma_m, 
              wmma_n, wmma_k, 1, 1>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_r_opt<wmma_m, wmma_n, wmma_k,  
              1, 1> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_r_opt<wmma_m, 
              wmma_n, wmma_k, 2, 1>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_r_opt<wmma_m, wmma_n, wmma_k,  
              2, 1> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }     else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_r_opt<wmma_m, 
              wmma_n, wmma_k, 1, 4>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_r_opt<wmma_m, wmma_n, wmma_k,  
              1, 4> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }else{
        printf("The warp_row/col_tiles is not supported within gemm \n");
      }
#endif

    }else{
      printf("Not use shm within gemm \n");
    }

  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds/runtimes);
  printf("TFLOPS: %.2f\n", (((double)M * N * K *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);



  return 0;

}



//  Default:
//  WaveRNN: gemm_O1/O3 gemm_O2/O4
//  SM num: 8 
//  beta = 0.0
int gemm_wmma_shm_persistent(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{

  // Default config.
  const int wmma_m = 16;
  const int wmma_n = 16;
  const int wmma_k = 16;

  // Default tile size: [16, 16, 16]
  // {M, N, K} % 16 == {0, 0, 0} 
  if((M % 16) || (N % 16) || (K % 16)){
    std::cout<<"M, N, K are illegal \n";
    return 0;
  }

  int M_tiles = M / 16;
  int N_tiles = N / 16;

  if(M_tiles * N_tiles < 16){
    std::cout<<"\n***** Not support now, TBD later ***** \n";
    return 0;
  }

  Configure cfg;
  cfg.wmma_m_ = 16;
  cfg.wmma_n_ = 16;
  cfg.wmma_k_ = 16;
  cfg.warp_size_ = 32;

  // Assume.  TBD.
  cfg.blk_tiles_ = 16;
  cfg.blk_warps_ = 8;

  // Assume
  // blockDim.x * blockDim.y == 128
  const int  chunk_col_ = 1024/16;  // specific.
  const int chunk_k_ = 8;

  if(M_tiles >= 4){
    cfg.blk_tiles_x_ = 8;
    cfg.blk_tiles_y_ = 8;
    cfg.warp_tiles_x_ = 8;
    cfg.warp_tiles_y_ = 1;
  }else if(M_tiles == 2){ 
    cfg.blk_tiles_x_ = 2; 
    cfg.blk_tiles_y_ = 8; //cfg.blk_tiles / cfg.blk_tiles_x;
    cfg.warp_tiles_x_ = 2;
    cfg.warp_tiles_y_ = 1;
  }else{
    cfg.blk_tiles_x_ = 1; 
    cfg.blk_tiles_y_ = 8;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 1 ;
  }

  cfg.blk_warps_x_ = cfg.blk_tiles_x_ / cfg.warp_tiles_x_;
  cfg.blk_warps_y_ = cfg.blk_tiles_y_ / cfg.warp_tiles_y_;

  int skew_half = 8;
  int shmem_sz = 0;
  bool USE_SHM = true;
  shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
        cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
      cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
      cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
      sizeof(half));
#if 1
  while(shmem_sz > 64 * 1024UL && USE_SHM){
    if(cfg.chunk_k_ >= 2){
      cfg.chunk_k_ /= 2;
      shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
            cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
          cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
          cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
          sizeof(half));
    }
    else
      USE_SHM = false;
  }
#endif

  //shmem_sz = 64 * 1024UL; 
#if 0
  printf("blk_row_tiles = %d, blk_col_tiles = %d \n", cfg.blk_tiles_y_,
      cfg.blk_tiles_x_);
  printf("warp_row_tiles = %d, warp_col_tiles = %d \n", cfg.warp_tiles_y_,
      cfg.warp_tiles_x_);
  printf("blk_row_warps = %d, blk_col_warps = %d \n", cfg.blk_warps_y_,
      cfg.blk_warps_x_);
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
  printf("chunk_k = %d \n", cfg.chunk_k_);
  shmem_sz = 32 * 1024UL; 
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
#endif
  shmem_sz = 64 * 1024UL; 
  int runtimes = 100;
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 grid(24, 1);
  dim3 block(256, 1, 1);

  for(int i = 0 ; i < runtimes ; i++){
    if(USE_SHM){
      if(cfg.warp_tiles_x_ == 8 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 8, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              8, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 4 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 4, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              4, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              2, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }

#if 0
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 2, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 2, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              2, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 4, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              2, 4, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 4, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 4, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }
#endif
      else{
        printf("The warp_row/col_tiles is not supported within gemm \n");
      }

    }else{
      printf("Not use shm within gemm \n");
    }
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds/runtimes);
  printf("TFLOPS: %.2f\n", (((double)M * N * K *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);



  return 0;

}



//  Default:
//  WaveRNN: gemm_O1/O3 gemm_O2/O4
//  SM num: 8 
//  beta = 0.0
int gemm_wmma_shm_splitk_persistent(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{

  // Default config.
  const int wmma_m = 16;
  const int wmma_n = 16;
  const int wmma_k = 16;

  // Default tile size: [16, 16, 16]
  // {M, N, K} % 16 == {0, 0, 0} 
  if((M % 16) || (N % 16) || (K % 16)){
    std::cout<<"M, N, K are illegal \n";
    return 0;
  }

  int M_tiles = M / 16;
  int N_tiles = N / 16;

  if(M_tiles * N_tiles < 16){
    std::cout<<"\n***** Not support now, TBD later ***** \n";
    return 0;
  }

  Configure cfg;
  //cfg.wmma_m_ = 16;
  //cfg.wmma_n_ = 16;
  //cfg.wmma_k_ = 16;
  //cfg.warp_size_ = 32;

  // Assume.  TBD.
  cfg.blk_tiles_ = 16;
  cfg.blk_warps_ = 8;

  // Assume
  // blockDim.x * blockDim.y == 128
  const int  chunk_col_ = (512/16);  // specific.
  const int chunk_k_ = 8;

  if(M_tiles >= 4){
    cfg.blk_tiles_x_ = 4;
    cfg.blk_tiles_y_ = 4;
    cfg.warp_tiles_x_ = 4;
    cfg.warp_tiles_y_ = 1;
  }else if(M_tiles == 2){ 
    cfg.blk_tiles_x_ = 2; 
    cfg.blk_tiles_y_ = 8; //cfg.blk_tiles / cfg.blk_tiles_x;
    cfg.warp_tiles_x_ = 2;
    cfg.warp_tiles_y_ = 1;
  }else{
    cfg.blk_tiles_x_ = 1; 
    cfg.blk_tiles_y_ = 8;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 1 ;
  }

  cfg.blk_warps_x_ = cfg.blk_tiles_x_ / cfg.warp_tiles_x_ ;
  cfg.blk_warps_y_ = cfg.blk_tiles_y_ / cfg.warp_tiles_y_ * 2;

  int skew_half = 8;
  int shmem_sz = 0;
  bool USE_SHM = true;
  shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
        cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
      cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
      cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
      sizeof(half));
#if 1
  while(shmem_sz > 64 * 1024UL && USE_SHM){
    if(cfg.chunk_k_ >= 2){
      cfg.chunk_k_ /= 2;
      shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
            cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
          cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
          cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
          sizeof(half));
    }
    else
      USE_SHM = false;
  }
#endif

  //shmem_sz = 64 * 1024UL; 
#if 0
  printf("blk_row_tiles = %d, blk_col_tiles = %d \n", cfg.blk_tiles_y_,
      cfg.blk_tiles_x_);
  printf("warp_row_tiles = %d, warp_col_tiles = %d \n", cfg.warp_tiles_y_,
      cfg.warp_tiles_x_);
  printf("blk_row_warps = %d, blk_col_warps = %d \n", cfg.blk_warps_y_,
      cfg.blk_warps_x_);
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
  printf("chunk_k = %d \n", cfg.chunk_k_);
  shmem_sz = 32 * 1024UL; 
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
#endif
  shmem_sz = 64 * 1024UL; 
  int runtimes = 100;
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 grid(16, 1);
  dim3 block(256, 1, 1);

  for(int i = 0 ; i < runtimes ; i++){
    if(USE_SHM){
        if(cfg.warp_tiles_x_ == 4 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 4, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              4, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              2, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              1, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
#if 0
      else if(cfg.warp_tiles_x_ == 8 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 8, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              8, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }

      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 2, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              1, 2, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              1, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              2, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 4, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              2, 4, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_splitk_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 4, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_splitk_persistent<wmma_m, wmma_n, wmma_k,  
              1, 4, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }
#endif
      else{
        printf("The warp_row/col_tiles is not supported within gemm \n");
      }

    }else{
      printf("Not use shm within gemm \n");
    }
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds/runtimes);
  printf("TFLOPS: %.2f\n", (((double)M * N * K *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);



  return 0;

}


//  Default:
//  WaveRNN: gemm_O1/O3 gemm_O2/O4
//  SM num: 8 
//  beta = 0.0
int gemm_wmma_shm_persistent_r(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{

  // Default config.
  const int wmma_m = 16;
  const int wmma_n = 16;
  const int wmma_k = 16;

  // Default tile size: [16, 16, 16]
  // {M, N, K} % 16 == {0, 0, 0} 
  if((M % 16) || (N % 16) || (K % 16)){
    std::cout<<"M, N, K are illegal \n";
    return 0;
  }

  int M_tiles = M / 16;
  int N_tiles = N / 16;

  if(M_tiles * N_tiles < 16){
    std::cout<<"\n***** Not support now, TBD later ***** \n";
    return 0;
  }

  Configure cfg;
  cfg.wmma_m_ = 16;
  cfg.wmma_n_ = 16;
  cfg.wmma_k_ = 16;
  cfg.warp_size_ = 32;

  // Assume.  TBD.
  cfg.blk_tiles_ = 16;
  cfg.blk_warps_ = 8;

  // Assume
  // blockDim.x * blockDim.y == 128
  const int  chunk_col_ = 1024/16;  // specific.
  const int chunk_k_ = 4;

  if(M_tiles >= 4){
    cfg.blk_tiles_x_ = 8;
    cfg.blk_tiles_y_ = 8;
    cfg.warp_tiles_x_ = 8;
    cfg.warp_tiles_y_ = 1;
  }else if(M_tiles == 2){ 
    cfg.blk_tiles_x_ = 2; 
    cfg.blk_tiles_y_ = 8; //cfg.blk_tiles / cfg.blk_tiles_x;
    cfg.warp_tiles_x_ = 2;
    cfg.warp_tiles_y_ = 1;
  }else{
    cfg.blk_tiles_x_ = 1; 
    cfg.blk_tiles_y_ = 8;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 1 ;
  }

  cfg.blk_warps_x_ = cfg.blk_tiles_x_ / cfg.warp_tiles_x_;
  cfg.blk_warps_y_ = cfg.blk_tiles_y_ / cfg.warp_tiles_y_;

  int skew_half = 8;
  int shmem_sz = 0;
  bool USE_SHM = true;
  shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
        cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
      cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
      cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
      sizeof(half));
#if 1
  while(shmem_sz > 64 * 1024UL && USE_SHM){
    if(cfg.chunk_k_ >= 2){
      cfg.chunk_k_ /= 2;
      shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
            cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
          cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
          cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
          sizeof(half));
    }
    else
      USE_SHM = false;
  }
#endif

  //shmem_sz = 64 * 1024UL; 
#if 0
  printf("blk_row_tiles = %d, blk_col_tiles = %d \n", cfg.blk_tiles_y_,
      cfg.blk_tiles_x_);
  printf("warp_row_tiles = %d, warp_col_tiles = %d \n", cfg.warp_tiles_y_,
      cfg.warp_tiles_x_);
  printf("blk_row_warps = %d, blk_col_warps = %d \n", cfg.blk_warps_y_,
      cfg.blk_warps_x_);
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
  printf("chunk_k = %d \n", cfg.chunk_k_);
  shmem_sz = 32 * 1024UL; 
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
#endif
  shmem_sz = 64 * 1024UL; 
  int runtimes = 100;
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 grid(24, 1);
  dim3 block(256, 1, 1);

  for(int i = 0 ; i < runtimes ; i++){
    if(USE_SHM){
      if(cfg.warp_tiles_x_ == 8 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent_r<wmma_m, 
              wmma_n, wmma_k, 8, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent_r<wmma_m, wmma_n, wmma_k,  
              8, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 4 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent_r<wmma_m, 
              wmma_n, wmma_k, 4, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent_r<wmma_m, wmma_n, wmma_k,  
              4, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent_r<wmma_m, 
              wmma_n, wmma_k, 2, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent_r<wmma_m, wmma_n, wmma_k,  
              2, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent_r<wmma_m, 
              wmma_n, wmma_k, 1, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent_r<wmma_m, wmma_n, wmma_k,  
              1, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }

#if 0
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 2, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 2, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              2, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 4, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              2, 4, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 4, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 4, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }
#endif
      else{
        printf("The warp_row/col_tiles is not supported within gemm \n");
      }

    }else{
      printf("Not use shm within gemm \n");
    }
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds/runtimes);
  printf("TFLOPS: %.2f\n", (((double)M * N * K *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);



  return 0;

}


//  Default:
//  WaveRNN: gemm_O1/O3 gemm_O2/O4
//  SM num: 8 
//  beta = 0.0
int gemm_wmma_shm_persistent_db(bool transa, bool transb, 
    int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{

  // Default config.
  const int wmma_m = 16;
  const int wmma_n = 16;
  const int wmma_k = 16;

  // Default tile size: [16, 16, 16]
  // {M, N, K} % 16 == {0, 0, 0} 
  if((M % 16) || (N % 16) || (K % 16)){
    std::cout<<"M, N, K are illegal \n";
    return 0;
  }

  int M_tiles = M / 16;
  int N_tiles = N / 16;

  Configure cfg;
  cfg.wmma_m_ = 16;
  cfg.wmma_n_ = 16;
  cfg.wmma_k_ = 16;
  cfg.warp_size_ = 32;

  // Assume.  TBD.
  cfg.blk_tiles_ = 16;
  cfg.blk_warps_ = 8;

  if(M_tiles * N_tiles < 16){
    std::cout<<"\n***** Not support now, TBD later ***** \n";
    return 0;
  }

  // Assume
  // blockDim.x * blockDim.y == 128
  const int  chunk_col_ = 1024/16;  // specific.
  const int chunk_k_ = 8;

  if(M_tiles >= 4){
    cfg.blk_tiles_x_ = 8;
    cfg.blk_tiles_y_ = 8;
    cfg.warp_tiles_x_ = 8;
    cfg.warp_tiles_y_ = 1;
  }else if(M_tiles == 2){ 
    cfg.blk_tiles_x_ = 2; 
    cfg.blk_tiles_y_ = 8; //cfg.blk_tiles / cfg.blk_tiles_x;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 2;
  }else{
    cfg.blk_tiles_x_ = 4; 
    cfg.blk_tiles_y_ = 4;
    cfg.warp_tiles_x_ = 1;
    cfg.warp_tiles_y_ = 2 ;
  }

  cfg.blk_warps_x_ = cfg.blk_tiles_x_ / cfg.warp_tiles_x_;
  cfg.blk_warps_y_ = cfg.blk_tiles_y_ / cfg.warp_tiles_y_;

  int skew_half = 8;
  int shmem_sz = 0;
  bool USE_SHM = true;
  shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
        cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
      cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
      cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
      sizeof(half));
#if 1
  while(shmem_sz > 64 * 1024UL && USE_SHM){
    if(cfg.chunk_k_ >= 2){
      cfg.chunk_k_ /= 2;
      shmem_sz = max(sizeof(half) * ((cfg.blk_tiles_x_ + cfg.blk_tiles_y_) * 
            cfg.wmma_m_) * (cfg.chunk_k_ * cfg.wmma_k_ + skew_half),
          cfg.wmma_m_ * (cfg.blk_warps_y_ * cfg.warp_tiles_y_) *
          cfg.wmma_n_ * (cfg.blk_warps_x_ * cfg.warp_tiles_x_) *
          sizeof(half));
    }
    else
      USE_SHM = false;
  }
#endif

  //shmem_sz = 64 * 1024UL; 
#if 0
  printf("blk_row_tiles = %d, blk_col_tiles = %d \n", cfg.blk_tiles_y_,
      cfg.blk_tiles_x_);
  printf("warp_row_tiles = %d, warp_col_tiles = %d \n", cfg.warp_tiles_y_,
      cfg.warp_tiles_x_);
  printf("blk_row_warps = %d, blk_col_warps = %d \n", cfg.blk_warps_y_,
      cfg.blk_warps_x_);
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
  printf("chunk_k = %d \n", cfg.chunk_k_);
  shmem_sz = 32 * 1024UL; 
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
#endif
  shmem_sz = 64 * 1024UL; 
  int runtimes = 100;
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 grid(16, 1);
  dim3 block(256, 1, 1);

  for(int i = 0 ; i < runtimes ; i++){
    if(USE_SHM){
      if(cfg.warp_tiles_x_ == 8 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent_db<wmma_m, 
              wmma_n, wmma_k, 8, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        //checkKernelErrors(( _gemm_wmma_shm_persistent_db<wmma_m, wmma_n, wmma_k,  
        checkKernelErrors(( _gemm_wmma_shm_persistent_db<wmma_m, wmma_n, wmma_k,  
              8, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 4 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent_db<wmma_m, 
              wmma_n, wmma_k, 4, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent_db<wmma_m, wmma_n, wmma_k,  
              4, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent_db<wmma_m, 
              wmma_n, wmma_k, 2, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent_db<wmma_m, wmma_n, wmma_k,  
              2, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent_db<wmma_m, 
              wmma_n, wmma_k, 1, 1, chunk_col_, chunk_k_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent_db<wmma_m, wmma_n, wmma_k,  
              1, 1, chunk_col_, chunk_k_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }

#if 0
      else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 2, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 2, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              2, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                //0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 4, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              2, 4, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 4){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 4, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              1, 4, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_,  //cfg.warp_size_,
                cfg.chunk_k_,
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc)));
      }
#endif
      else{
        printf("The warp_row/col_tiles is not supported within gemm \n");
      }

    }else{
      printf("Not use shm within gemm \n");
    }
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds/runtimes);
  printf("TFLOPS: %.2f\n", (((double)M * N * K *
          2)/(milliseconds / 1e3)) / 1e12 * runtimes);



  return 0;

}




// OP_N: 0 OP_T: 1.
// lda = K/M, ldb = K/N, ldc = N
int gemm_wmma_half(bool OP_A_N, bool OP_B_N, int M, int N, int K,
    float alpha,
    half *A, int lda,
    half *B, int ldb,
    float beta,
    half *C, int ldc)
{

  const int WMMA_M = 16;  
  const int WMMA_N = 16; 
  //const int WMMA_K = 16; 

  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  //printf("grid_x = %d, grid_y = %d, blk_x = %d, blk_y = %d \n", gridDim.x, 
  //    gridDim.y, blockDim.x, blockDim.y);


  checkKernelErrors((_gemm_wmma_half<<<gridDim, blockDim>>>(OP_A_N, OP_B_N, A, 
          B, C, C, M, N, K, alpha, beta,lda, ldb, ldc)));

  return 0;

}

