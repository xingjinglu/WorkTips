import test
import time

start = time.time()
sz = test.sizeyunsuan();
a = 22
b = 13
sub = sz.sub(a, b)
end = time.time()
total = (end - start) *  1000.0
print(f"{a} - {b} = {sub}, time = {total} ms")
