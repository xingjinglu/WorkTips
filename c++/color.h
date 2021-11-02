#ifndef _COLOR_H
#define _COLOR_H

#if 0
enum Color {Cred, Cblue, Cgreen};
#endif

#define COLORS \
  X(Cred, "red") \
  X(Cblue, "blue")\
  X(Cgreen, "green")

#define X(a, b) a,
enum Color { COLORS };
#undef X

#endif

