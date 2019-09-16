#include <stdio.h>

#include "color.h"

//static char *ColorStrings[] = {"red", "blue", "green"};


int main()
{
#if 0
  enum Color c = Cgreen;
  printf("the color is %s \n", ColorStrings[c]);
#endif

#define X(a, b) b,
  static char * ColorStrings[] = {COLORS};
#undef X
  enum Color c = Cgreen;
  printf("the color is %s \n", ColorStrings[c]);

  return 0;
}

