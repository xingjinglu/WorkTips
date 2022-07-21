#include<iostream>
#include <functional>

void call(int arg, std::function<void(int) > func){
  func(arg);
}

void call2(std::function<void()> func){
  func();
}

int main()
{

  
  //
  auto printSquare = [] (int x) { std::cout << x * x << std::endl; };
  call(2, printSquare);

  // 
  auto a = [](int x, int y) -> decltype(x)
  {
    return x + y;
  };
  std::cout << a(10, 12) << std::endl; 

  // 
  int i = 3;
  auto print2 = [i]() { std::cout<< i * i << std::endl;}; // copy
  call2(print2);


  //
  int c = 4;
  [&c](int x) { c += x; }(2);
  std::cout << "c = " << c << std::endl;

  return 0;
}
