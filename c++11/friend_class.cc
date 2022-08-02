#include "friend_class.h"
#include<iostream>


int main()
{
  Base b(2, 3);
  b.print();

  Cfriend c;
  c.test(b);
  b.print();


  return 0;
}
