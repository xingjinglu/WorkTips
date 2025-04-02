def simple_generator():
    yield 1
    sum = 1 + 2
    print(sum)
    yield 2
    sum = 2 + 3
    print(sum)
    yield 3
    sum = 4 + 3
    print(sum)

gen = simple_generator()
print(next(gen))  # 输出: 1
print(next(gen))  # 输出: 2
print(next(gen))  # 输出: 3
# print(next(gen))  # 如果再调用一次，会引发 StopIteration 异常
