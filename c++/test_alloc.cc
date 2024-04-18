#include<iostream>

int main()
{
  int n = 10;

  if (n > std::size_t(-1) / sizeof(float)) 
    std::cout<<"error"<<std::endl;
  std::cout<<std::size_t(-1)/sizeof(float) << std::endl;
  return 0;
}
