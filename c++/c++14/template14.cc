#include<iostream>

// reference.
template<typename T>
T addT(T a, T b)
{

  return (a+b);
}


// Universal reference.


// Array parameter.
template<typename T>
void reduceArray(T param)
{


  return;
}

// Array parameter.
template<typename T>
void reduceArray1(T& param)
{


  return;
}

// Array parameter.
template<typename T>
void reduceArray2(T* param)
{

  return;
}

// Array parameter.
template<typename T>
void reduceArray3(T&& param)
{

  return;
}


int main()
{

  int a = 10, b = 20;
  int c = addT(a, b);
  std::cout<<"c = " << c << std::endl;

  float a0 = 10.3, b0 = 20.5;
  float c0 = addT<float>(a0, b0);
  std::cout<<"c0 = " << c0  << std::endl;

  short a1 = 12, b1 = 15;
  short c1 = addT(a1, b1);
  std::cout<<"c1 = " << c1 << std::endl;

  // Array.
  const char name[] = "J. P. Briggs";
  const char *ptrtoName = name;
  reduceArray(name); // T -> const char*

  reduceArray1(name); // T -> char const [13]
                      //
  reduceArray2(name); // T -> char const
                      //
  reduceArray3(name); // T -> char const (&) [13]  ???
                      //


  return 0;
}

