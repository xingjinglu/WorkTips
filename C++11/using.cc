#include "helper.h"

template<typename T>
using smap = std::map<std::string, T>;

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

  smap<int> mymap{{"one", 1}, {"two", 2}};
  for(const auto &p:mymap)
    std::cout<<p.first<<", " << p.second << std::endl;

return 0;
}
