#include<climits>
#include<cstdio>

int main()
{
  long long min = LLONG_MIN;
  long long max = LLONG_MAX;
  unsigned long long umax = ULONG_MAX;

  printf("long long min = %lld \n", min);
  printf("long long max = %lld \n", max);
  printf("long long umax = %llu \n", umax);


  return 0;
}

