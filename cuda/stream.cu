#include<cuda.h>
#include<stdio.h>

__global__ void VecAdd(float *A, float *B, float *C)
{
  int i = threadIdx.x;
  for(int j = 0; j < 1000; j++)
  C[i] = A[i] + B[i];
}

__global__ void VecMul(float *A, float *B, float *C)
{
  int i = threadIdx.x;
  for(int j = 0; j < 1000; j++)
  C[i] = A[i] * B[i];
}



int main()
{
  int N = 1024 * 10;
  dim3 threadPerBlock(256);
  dim3 numBlocks(N/threadPerBlock.x);

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  //cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  //cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

  // Host memory.
  float *hA, *hB, *hC, *hAMul, *hBMul, *hCMul;
#if 0
  hA = (float*) malloc (sizeof(float) * N);
  hB = (float*) malloc (sizeof(float) * N);
  hC = (float*) malloc (sizeof(float) * N);
  hAMul = (float*) malloc (sizeof(float) * N);
  hBMul = (float*) malloc (sizeof(float) * N);
  hCMul = (float*) malloc (sizeof(float) * N);
#endif
  cudaMallocHost((void**)&hA, sizeof(float) * N); 
  cudaMallocHost((void**)&hB, sizeof(float) * N); 
  cudaMallocHost((void**)&hC, sizeof(float) * N); 
  cudaMallocHost((void**)&hAMul, sizeof(float) * N); 
  cudaMallocHost((void**)&hBMul, sizeof(float) * N); 
  cudaMallocHost((void**)&hCMul, sizeof(float) * N); 
  for(int i = 0; i < N; i++){
    hA[i] = i;
    hB[i] = i + 1;
    hC[i] = 0.0f;
    hAMul[i] = i % 10;
    hBMul[i] = i % 10;
    hCMul[i] = 0.0f;

  }

  // Device memory.
  float *A, *B, *C, *AMul, *BMul, *CMul;
  cudaMalloc((void**)&A, sizeof(float) * N); 
  cudaMalloc((void**)&B, sizeof(float) * N); 
  cudaMalloc((void**)&C, sizeof(float) * N); 
  cudaMalloc((void**)&AMul, sizeof(float) * N); 
  cudaMalloc((void**)&BMul, sizeof(float) * N); 
  cudaMalloc((void**)&CMul, sizeof(float) * N); 

  // Host->Device
  cudaMemcpyAsync(A, hA, sizeof(float)*N, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(B, hB, sizeof(float)*N, cudaMemcpyHostToDevice, stream1);
  //cudaMemcpyAsync(C, hC, sizeof(float)*N, cudaMemcpyHostToDevice, stream1);
  VecAdd<<<numBlocks, threadPerBlock, 0, stream1>>>(A, B, C);

  cudaMemcpyAsync(AMul, hAMul, sizeof(float)*N, cudaMemcpyHostToDevice, stream2);
  cudaMemcpyAsync(BMul, hBMul, sizeof(float)*N, cudaMemcpyHostToDevice, stream2);
  VecMul<<<numBlocks, threadPerBlock, 0, stream2>>>(AMul, BMul, CMul);
  cudaMemcpyAsync(hC, C, sizeof(float)*N, cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(hCMul, CMul, sizeof(float)*N, cudaMemcpyDeviceToHost, stream2);

  //cudaMemcpyAsync(BMul, hBMul, sizeof(float)*N, cudaMemcpyHostToDevice, stream2);
  //cudaMemcpyAsync(CMul, hCMul, sizeof(float)*N, cudaMemcpyHostToDevice, stream2);
  //cudaMemcpyAsync(hCMul, CMul, sizeof(float)*N, cudaMemcpyDeviceToDevice, stream2);
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  printf("hC = %f, hCMul = %f \n", hC[10], hCMul[12]);
  return 0;
}
