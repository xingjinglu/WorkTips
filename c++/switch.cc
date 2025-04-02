#include<iostream>

int main()
{

  int a = 10;
  switch (a){
    case 10:
      std::cout<<"10"<<std::endl;
    case 9:
      std::cout<<"9"<<std::endl;

    default:
      std::cout<<"0" << std::endl;
  }

  return 0;
}
