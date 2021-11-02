#include<stdio.h>
#include"cmult.h"
void print_msg(const char *s)
{
  printf("%s\n", s);
}


int cmult(int a, int b)
{
  int sum = 0.0;
  sum = a + b;
  return sum;
}

float add_float(float a, float b)
{
  float sum = a + b;
  return sum;
}


float vec_add_float(float *a, float *b, float *c, int num)
{

  int i = 0;
  for(i = 0; i < num; i++){
    c[i] = a[i] + b[i];
  }

  return 0.0;
}

