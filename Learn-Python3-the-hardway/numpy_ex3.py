import numpy as np

#
print(np.intc)
print(np.float)


# 
a = np.array([1, 2, 3], dtype='float')
print(a)

b = np.array([1.2, 2.4, 3.6])
print(b)
print(b.astype(int))
print(b.astype(float))

c = np.dtype(int)
print(np.issubdtype(c, np.floating))
c = np.dtype(float)
print(np.issubdtype(c, np.floating))


d = np.array([[1, 2], [3, 4], (1+1j, 3)])
print(d)

e = np.zeros((2, 3))
print(e)

f = np.linspace(1., 4, 6)
print(f)


