#include<iostream>

int main()
{
  int a = 1 + 2;
  a += 4;

  int &&b = 1 + 2;
  b += 4;
  std::cout<<"a = " << a <<", b = " << b << std::endl;


  return 0;
}
