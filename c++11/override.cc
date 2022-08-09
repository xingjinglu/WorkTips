#include "override.h"
#include <iostream>

double Circle::pi_ = 3.14;

int main()
{
  Rect rt(3.1, 4.2);
  float area1 = rt.getArea();
  std::cout<<"rt.area: " << area1 << std::endl;

  Triangle tr = Triangle(3.1, 4.2);
  float area2 = tr.getArea();
  std::cout<<"tr.area: " << area2 << std::endl;


  // upcast. 
  Rect &rt2 = tr;
  float area3 = rt2.getArea(); 
  std::cout<<"rt2.area: " << area3 << std::endl;

  //
//Circle::pi = 3.14;
Circle c(10);
double area4 = c.getArea();
std::cout<<"Circle.area: " << area4 << std::endl;



  return 0;
}
