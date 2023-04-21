
from collections import namedtuple

# fluent python ch2.
#

def test():

    # list

    #
    a = [x for x in range(10)]
    print(a)

    #
    str = "chatgpt is product of openai"
    b = [ord(x) for x in str]
    print(b)

    # 推导
    colors = ['black', 'white']
    sizes = ['S', 'M', 'L']
    tshirts = [(color, size) for color in colors for size in sizes]
    tshirts = [(size, color) for color in colors 
               for size in sizes]
    print(tshirts)

    # 生成器
    tshirts = ('%s %s' % (c, s) for c in colors for s in sizes)
    for tshirt in tshirts:
        print(tshirt)

    # 元组  tuple
    a = 20
    b = 10
    a, b = b, a
    print(a, " ", b)

    a,*b,c =  range(5)
    print( *b)

    # named tuple
    student = namedtuple('student', 'name age sex')
    rudy = student('rudy', '8', 'boy')
    print(rudy)

    City = namedtuple('City', 'name country population coordinates')
    tokyo = City('Tokyo', 'JP', 36.933, (35.689722, 139.691667))
    print(tokyo)

    str = "fluent python is a good book"
    print(str[:4])
    print(str[10:14])

    #
    board = [['_'] * 3 for i in range(3)]
    print(board)

    # sort
    fruits = ['grape', 'raspberry', 'apple', 'banana']
    s_fruits = sorted(fruits)
    print("fruits: ", fruits)
    print(s_fruits)
    fruits.sort()
    print("fruits: ", fruits)



if __name__ == "__main__":
    test()
