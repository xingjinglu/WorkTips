#include<iostream>

int main()
{
  // c++11
  decltype(3) b = 3;
  std::cout<<"b = " << b << std::endl;

  // c++14
  decltype(auto) c = 3;
  std::cout<<"c = " << c << std::endl;


  return 0;
}
