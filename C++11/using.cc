#include "helper.h"

int main()
{
  using int32 = int;
  int32 a;
  a  = 23;
  std::cout<<"a : " << a <<"\n";

  typedef int int32;
  int32 b;
  b  = 24;
  std::cout<<"b : " << b<< "\n";

return 0;
}
