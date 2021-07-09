#ifndef __CMULT_H
#define __CMULT_H
extern "C" {
  void print_msg(const char *s);
  int cmult(int int_param, int float_param);
  float add_float(float a, float b);
  //float vec_add_float(float &a, float &b, float &c, int num);
  float vec_add_float(float *a, float *b, float *c, int num);
}
#endif
