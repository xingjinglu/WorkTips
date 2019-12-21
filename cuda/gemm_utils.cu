#include "gemm_utils.h"

void  checkKernelsErrors(const char *prefix, const char *postfix) 
{       
  cudaDeviceSynchronize();
  if(cudaPeekAtLastError() != cudaSuccess){                                
    printf("\n%s Line %d: %s %s\n", prefix, __LINE__, 
        cudaGetErrorString(cudaGetLastError()),   
        postfix);                                                          
    cudaDeviceReset();                                                     
    exit(1);                                                               
  }                                                                        

  return;
} 
