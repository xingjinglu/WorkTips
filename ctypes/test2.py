from ctypes import *

class Bottles:
    def __init__(self, number):
        self._as_parameter_ = number


bottles = Bottles(42)
print(bottles)
print("%d bottles of beer\n", bottles)
