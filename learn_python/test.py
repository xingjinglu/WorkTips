class sizeyunsuan:
    '''
    这里可以写类的说明
    '''

    # 默认构造函数
    def __init__(self):
        pass        # 空函数

    # self: 必须带的参数，指向类的对象
    # a, b: 普通参数，与非类 成员函数一样
    def add(self, a = 0, b = 0):
        res = a + b
        return res

    def sub(self, a = 0, b = 0):
        res = a - b
        return res


if __name__ == "__main__":
    sz = sizeyunsuan()     # 声明类的对象, sz
    sum = sz.add(11, 12)   # 调用类(sizeyunsuan)的成员函数  add
    print(sum)

    

