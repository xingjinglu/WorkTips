#include<iostream>
#include<cassert>
#include<stdlib.h>

template<typename T, typename U> int bit_copy(T &a, U &b)
{
  static_assert(sizeof(a) == sizeof(b), "bit copy operands must have the same bit width");

  return 0;
}

int main()
{
#if __cplusplus < 201103L
#error "should use C++ 11 implementation"
#endif

  int a = 10;
  int b = 12 ;
  bit_copy<int, int>(a, b);

  return 0;
}
