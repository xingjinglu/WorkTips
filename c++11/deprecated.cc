#include <iostream>

int main()
{

  // tip 1.
  char *str = "is this ok";
  const char *cstr = "are you ok"; 
  auto astr = "is auto better"; 

  std::cout<< str << std::endl;
  std::cout<< cstr << std::endl;
  std::cout << astr << std::endl;
  return 0;
}
