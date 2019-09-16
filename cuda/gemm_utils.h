#ifndef _GEMM_UTILS_H
#define _GEMM_UTILS_H

#include<cstdio>
#include<iostream>
#include<sstream>
#include<assert.h>

void  checkKernelsErrors(const char *prefix, const char *postfix);

#define checkCudaErrors(expr) do {                                                      \
  expr;                                                                                  \
  cudaError_t __err = cudaGetLastError();                                                 \
  if (__err != cudaSuccess) {                                                             \
    printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));    \
    abort();                                                                              \
  }                                                                                       \
} while(0)

#define checkSyncKernelErrors(expr) do {                                                      \
  expr;                                                                                   \
  cudaDeviceSynchronize();                                                                \
  cudaError_t __err = cudaGetLastError();                                                 \
  if (__err != cudaSuccess) {                                                             \
    printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));    \
    abort();                                                                              \
  }                                                                                       \
} while(0)


#define checkKernelErrors(expr) do {                                                      \
  expr;                                                                                   \
  cudaError_t __err = cudaGetLastError();                                                 \
  if (__err != cudaSuccess) {                                                             \
    printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));    \
    abort();                                                                              \
  }                                                                                       \
} while(0)


#if 0
#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)                                                                                                      


#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)                                                                                                      

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)                                                                                                      

#define CU_SAFE_CALL(fun)                                              \
{                                                                      \
  int ret;                                                             \
  std::stringstream _error;                                            \
  if ((ret = (fun)) != 0) {                                            \
    _error << "cudaError_t " << ret << " : \""                         \
           << cudaGetErrorString((cudaError_t)ret)                     \
    << "\" returned from '" << #fun << "'";                            \
    FatalError(_error.str());                                          \
  }                                                                    \
}                    

#define CU_SAFE_CALL(fun)                                              \
{                                                                      \
  int ret;                                                             \
  std::stringstream _error;                                            \
  if ((ret = (fun)) != 0) {                                            \
    _error << "cudaError_t " << ret << " : \""                         \
           << cudaGetErrorString((cudaError_t)ret)                     \
    << "\" returned from '" << #fun << "'";                            \
    FatalError(_error.str());                                          \
  }                                                                    \
} 


#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)

#endif




#endif
