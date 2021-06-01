from ctypes import *
import ctypes
import os

sotest = cdll.LoadLibrary(os.getcwd() + "/libcmult.so")
sotest.print_msg("hello, my shared object used by python!")

print("4+5=%s" %sotest.cmult(4, 5))

x, y = 5.2, 6.4
sotest.add_float.restype = ctypes.c_float
sotest.add_float.argtypes = (ctypes.c_float, ctypes.c_float)
#result = sotest.add_float(ctypes.c_float(x), ctypes.c_float(y));
result = sotest.add_float(x, y);
print(f"In Python: float: {x} float {y:.1f} return val {result:.1f}")



