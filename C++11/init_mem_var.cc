#include<iostream>
#include "init_mem_var.h"

Conv::Conv():kernelh(0), kernelw(0), stridex(0), stridey(0)
{
}
Conv::~Conv()
{
}

ConvCpp11::ConvCpp11()
{}
ConvCpp11::~ConvCpp11()
{}

// Work with C++98.

int main()
{
  Conv c1;
  ConvCpp11 c2;

  std::cout<<c1.stridex <<", " << c2.stridex << std::endl;

  return 0;
}
