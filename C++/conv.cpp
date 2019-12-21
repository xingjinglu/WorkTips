#include<iostream>


int main()
{
  int M = 256, N = 256;
  float *Img = new float[M * N];
  float *Dst = new float[M * N];
  float Mask[9] = {0.4, 0.2, 0.4, 0.2, 0.4, 0.4, 0.4, 0.4, 0.2};

  for(int i = 1; i < M - 1; i++)
    for(int j = 1; j < N - 1; j++){
      Dst[i * N  + j] = Img[(i - 1) * N + (j - 1)] * 0.4 + Img[(i - 1) * N + j] * 0.2 + 
                        Img[(i - 1) * N + (j + 1)] * 0.4 + Img[i * N + (j - 1)] * 0.2 + 
                        Img[i * N + j] * 0.4 + Img[i * N + (j + 1)] * 0.4 + 
                        Img[(i + 1) * N + (j - 1)] * 0.4 + Img[(i + 1) * N + j] * 0.2 + 
                        Img[(i + 1) * N + (j + 1)] * 0.4;
    }
  return Dst[11];
}
