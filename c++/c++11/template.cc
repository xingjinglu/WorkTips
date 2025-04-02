#include<iostream>

template<class T>
T add(T &a, T &b)
{
  return (a+b);
}

template<typename T>
T sub(T &a, T &b)
{
  return (a - b);
}


int main()
{
  int c, a, b, d;
  a = 10;
  b = 12;
  c = add<int>(a, b);
  d = sub<int>(a, b);

  std::cout<<"a + b: " << c << std::endl;
  std::cout<<"a - b: " << d << std::endl;


  return 0;
}
