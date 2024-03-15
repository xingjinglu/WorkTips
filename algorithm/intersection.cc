#include <iostream>
#include <vector>
#include <map>

// **

void do_cpp17()
{
  std::vector< std::vector<int> > a{
    {3, 4, 5, 6, 8},
      {0, 2, 3, 4, 5, 8},
      {0, 1, 4, 8},
      {1, 2, 3, 4, 5, 6, 7, 8}
  };

  std::multimap<int, int> s;
  s.insert(std::pair<int, int>(a[0].size(), 0));
  s.insert(std::pair<int, int>(a[1].size(), 1));
  s.insert(std::pair<int, int>(a[2].size(), 2));
  s.insert(std::pair<int, int>(a[3].size(), 3));

  for(auto &[key, val]:s)
    std::cout<< key << " ==> " << val << std::endl;


  return;
}

int main()
{
  do_cpp17();
  
  return 0;
}
