#include <cuda_runtime.h>
#include <cuda.h>

__device__ int ptr=0;

__global__ void a()
{
    int b[100];
  //atomicAdd(&ptr,1);
  b[0]=ptr;
#pragma unroll
  for(int i=1; i<200; i++)
  {
    //      for(int j=1;j<90;j++)
    {
      //b[i][j]=b[i-1][j-1]+1;
      b[i] = b[i-1]+1;
    }
  }
  ptr=b[7]+1;
}

int main()
{
  a<<<1,1>>>();
  return 0;
}
