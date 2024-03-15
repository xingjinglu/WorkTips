#include<iostream>


template<int N>
constexpr int fibonacci()
{
  if constexpr(N >= 2)
    return fibonacci<N-1>() + fibonacci<N-2>();
  else
    return N;
}


int main()
{
  int a = fibonacci<10>();
  std::cout<<"a = " << a << std::endl;
  return 0;
}
