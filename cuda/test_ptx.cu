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



