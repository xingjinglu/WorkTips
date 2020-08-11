#include<stdio.h>

int f(int n)
{
  if(n == 1) return 1;
  else if(n == 2) return 2;
  else 
    return (f(n-1) + f(n-2));
}

int top_down(int n)
{
  int sum = f(n);
  printf("total  = %d \n", sum);

  return sum;
}


int bottom_up(int n)
{


  return 0;
}

// 10 step, 1 step or 2 steops evry time, how many  
int main()
{
  top_down(10);


  return 0;
}
