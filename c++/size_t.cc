#include<iostream>
#include<stdlib.h>


int main()
{
  std::cout<<"sizeof(int) = " << sizeof(int) << std::endl;
  std::cout<<"sizeof(float) = " << sizeof(float) << std::endl;
  std::cout<<"sizeof(double) = " << sizeof(double) << std::endl;
  std::cout<<"sizeof(void**) = " << sizeof(void**) << std::endl;
  std::cout<<"sizeof(size_t) = " << sizeof(size_t) << std::endl;
  std::cout<<"sizeof(long) = " << sizeof(long long) << std::endl;

  int *ptr =static_cast<int*>( malloc(10));
  std::cout<<"ptr="<<ptr<<std::endl;
  std::cout<<"&ptr[0] = " << &ptr[0] << std::endl;
  std::cout<<"&ptr[-1] = " << &ptr[-1] << std::endl;

  return 0;
}


