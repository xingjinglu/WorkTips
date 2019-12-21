#include "helper.h"

int main()
{
  int status  = -4;

  auto a = 13.4;
  std::cout<<"a = " << a << "\n";
  std::cout<<"a has type: " << abi::__cxa_demangle(typeid(a).name(), NULL, NULL, &status) << std::endl;

  decltype(a) b;
  std::cin>>b;
  std::cout<<"b = " << b <<"\n";
  std::cout<<"b has type: " << abi::__cxa_demangle(typeid(b).name(), NULL, NULL, &status) << std::endl;


  return 0;
}
