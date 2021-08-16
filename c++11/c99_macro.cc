#include<iostream>
#include<stdio.h>

#define PR(...) printf(__VA_ARGS__)
#define DEBUG(...) { \
  fprintf(stderr, "%s: Line %d \t", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\n"); \
}



const char * func_str()
{
  return __func__;
}

int main()
{

#if __STDC_HOSTED__
  std::cout<<"There is standard c library \n";
#endif

  std::cout<<"STDC: " << __STDC__ << "\n";

  //std::cout<<"c version: " << __STDC_VERSION__ <<"\n";
  //
  //std::cout<<"ISO/IEC: " << __STDC_ISO_10646__ << "\n";


  std::cout<<"__func__: " << func_str() << std::endl;

  int a = 10;
  PR("a = %d \n", a);

  DEBUG("log error");
  
  return 0;
}
