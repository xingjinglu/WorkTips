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

#if 1
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
#if 0
          if(blockIdx.x == 0 && tId == 0){
            for(int i = 0; i < 64; i++){
              printf("b[%d] = %f \n", i, __half2float(shm[i]));
            }
          }
#endif
        
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

  for(int run = 0; run < 1; run++){
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
               // wmma::mma_sync(acc[i][j], a[i], b[j][tile_k + k_step], acc[i][j]);
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

  Configure cfg;
  cfg.wmma_m_ = 16;
  cfg.wmma_n_ = 16;
  cfg.wmma_k_ = 16;
  cfg.warp_size_ = 32;
  cfg.blk_warps_ = 4;

  // Assume.  TBD.
  cfg.blk_tiles_ = 16;

  if(M_tiles * N_tiles < 16){
    std::cout<<"\n***** Not support now, TBD later ***** \n";
    return 0;
  }

  // Assume
  // blockDim.x * blockDim.y == 128
  const int  chunk_col_ = 1024/16;  // specific.
  cfg.chunk_k_ = 4;

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
  //cfg.blk_warps_y_ = cfg.blk_tiles_y_ / cfg.warp_tiles_y_;

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
  shmem_sz = 64 * 1024UL; 
  printf("Required shared memory size: %lu Kb\n", shmem_sz / 1024UL);
#endif
  int runtimes = 100;
  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));    
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start));

  dim3 grid(32, 1);
  dim3 block(256, 1, 1);

  for(int i = 0 ; i < runtimes ; i++){
    if(USE_SHM){
      if(cfg.warp_tiles_x_ == 8 && cfg.warp_tiles_y_ == 1){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 8, 1, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkSyncKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              8, 1, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                cfg.chunk_k_,
                // 0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }
#if 0
      else if(cfg.warp_tiles_x_ == 2 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 2, 2, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkSyncKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
              2, 2, chunk_col_> 
              <<<grid, block, shmem_sz>>>(
                cfg.blk_warps_y_, cfg.blk_warps_x_,
                cfg.blk_warps_, //cfg.warp_size_,
                cfg.chunk_k_,
                // 0, 0, 
                M, N, K, 
                alpha, A, lda, B, ldb, beta, C, ldc))
            );
      }else if(cfg.warp_tiles_x_ == 1 && cfg.warp_tiles_y_ == 2){
        checkCudaErrors(cudaFuncSetAttribute(_gemm_wmma_shm_persistent<wmma_m, 
              wmma_n, wmma_k, 1, 2, chunk_col_>, 
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_sz));
        checkSyncKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
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
        checkSyncKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
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
        checkSyncKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
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
        checkSyncKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
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
        checkSyncKernelErrors(( _gemm_wmma_shm_persistent<wmma_m, wmma_n, wmma_k,  
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


