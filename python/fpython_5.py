

def factorial(n):
    ''' return n! '''
    if n < 2:
        return 1
    else:
        return n * factorial(n-1)

import random

class BingoCage:
    def __init__(self, items):
        self._items = list(items)
        random.shuffle(self._items)

    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage')


    def __call__(self):
        return self.pick()

def tag(name, *content, cls=None, **attrs):
    """generate one or many HTML labels"""
    return "hello"

import bobo

@bobo.query('/')
def hello(person):
    return 'Hello {}'.format(person)

def test():

    # 5.1 first-class function
    print("function")
    print("factorial.__doc__: ", factorial.__doc__)

    fact =  factorial
    print(fact)
    print(fact(5))

    # map() function,
    # map() function returns a map object(which is an iterator) of the results 
    # after applying the given function to each item of a given iterable 
    # (list, tuple etc.)
    a = map(factorial, range(6))
    print("map(factorial,range(6)): ", a)
    a = list(map(factorial, range(6)))
    print("list(map(factorial,range(6))): ", a)

    # 5.2 higher function: map, filter, reduce , apply
    a = list(map(factorial, filter(lambda n: n % 2, range(6))))
    print("lambda:", a)

    # 1) python3.0: map and filter返回生成器（一种迭代器），因此可以用生成器表达式
    # 替代
    a = list(map(factorial, range(4)))
    print(a)
    a = list(factorial(n) for n in range(4))
    print(a)
    # 2) reduce: 被sum取代
    # 3) all and any 也是内置的归约函数
    # all(iteratable) any(iteratable)


    # 5.3 匿名函数
    # lambda 只能使用纯表达式，即不能赋值、while、try等语句
    a = list(map(factorial, filter(lambda n: n % 2, range(4))))
    print(a)
    a = list(factorial(n) for n in range(4) if n % 2)
    print(a)

    # builtin function, sum, any, all



    #  5.4 可调用对象
    '''
    1. 用户定义函数: def/lambda创建
    2. 内置函数: 用c语言实现的函数，len
    3. 内置方法：用c语言实现的方法
    4. 方法：在类的定义体重定义的函数
    5. 类: 会调用__new__方法创建一个实例，并运行__init__方法初始化实例，最后把实例
       返回给调用方
    6. 类的实例：如果类定义了__call__,那么它的实例可以作为函数调用，实际执行__call__
    7. 生成器函数：使用yield关键字的函数或方法, 调用生成器函数返回的是生成器对象
  '''

    # 5.5 用户定义的可调用类型
    '''
    任何定义__call__()方法的python对象都可以表现的像函数

    '''
    bingo = BingoCage(range(3)) 
    print("bingo(): ", bingo())

    # 5.6 函数内省
    print(dir(factorial))

    # 5.7 从定位参数到仅限关键字参数  
    '''
    位置参数
    关键字参数/默认值参数
    可变参数,func(a, b, c, *var), var是一个元组
    可变关键字参数: **keyvar,接受0个或者多个关键字参数，以字典的形式传入函数体
    仅限关键字参数：只能传入关键字参数；可变参数后面的关键字参数都是仅限关键字参数；
        单个*表示不接受任何可变参数，可以表示普通参数结束。
    '''

    # 5.8
    # 5.9 





if __name__ == "__main__":
    test()

