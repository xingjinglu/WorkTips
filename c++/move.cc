#include<iostream>

int main()
{
  int a = 10, b = 20;
  a = std::move(b);
  std::cout<<"a = " << a << ", b = " << b << std::endl;
  return 0;
}
