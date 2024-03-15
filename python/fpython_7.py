
#ch7. 函数装饰器和闭包

def deco(func):
    def inner():
        print("running inner()")
    return inner


@deco
def target():
    print('running target()')

def target2():
    print('running target()')


class Averager():

    def __init__(self):
        self.series = []

    def __call__(self, new_value):
        self.series.append(new_value)
        total = sum(self.series)
        return total/len(self.series)

def make_averager():
    series = []

    def averager(new_value):
        series.append(new_value)
        total = sum(series)
        return total/len(series)

    return averager

def test():
    # 7.1 装饰器基础知识
    """

    """
    target()
    target2()
    print("target: {} ".format(target))

    # 7.5 闭包
    avg = Averager()
    a = avg(10)
    print("avg: {} ".format(a))
    a = avg(11)
    print("avg: {} ".format(a))
    a = avg(12)
    print("avg: {} ".format(a))
    
    #
    avg2 = make_averager()
    b = avg2(10)
    b = avg2(11)
    b = avg2(12)
    print("b: ", b)
    print(avg2.__code__.co_varnames)
    print(avg2.__code__.co_freevars)

    # 7.6 nonlocal 声明


    print("hello")


if __name__ == "__main__":
    test()
