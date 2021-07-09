from ctypes import *
import ctypes
import os

c_float_p = ctypes.POINTER(ctypes.c_float)
c_float = ctypes.c_float
c_int = ctypes.c_int

sotest = cdll.LoadLibrary(os.getcwd() + "/libcmult.so")
sotest.print_msg("hello, my shared object used by python!")

print("4+5=%s" %sotest.cmult(4, 5))

x, y = 5.2, 6.4
sotest.add_float.restype = ctypes.c_float
sotest.add_float.argtypes = (ctypes.c_float, ctypes.c_float)
#result = sotest.add_float(ctypes.c_float(x), ctypes.c_float(y));
result = sotest.add_float(x, y);
print(f"In Python: float: {x} float {y:.1f} return val {result:.1f}")


# float array.
#sotest.vec_add_float.argtypes(ctypes.pointer(c_float), ctypes.pointer(c_float), ctypes.pointer(c_float), c_int)
#sotest.vec_add_float.argtypes(c_float_p, c_float_p, c_float_p, c_int)
sotest.vec_add_float.restypes = c_float

size = 10
a = (ctypes.c_float * size)(1, 2, 3)
b = (ctypes.c_float * size)(2, 4, 4)
c = (ctypes.c_float * size)()

res = sotest.vec_add_float(ctypes.cast(a, c_float_p ),
                           ctypes.cast(b, c_float_p),
                           ctypes.cast(c, c_float_p), size)
for i in range(size):
    print(a[i], b[i], c[i])

# multi-dims array.
a = ((c_int * 4) * 3)()



