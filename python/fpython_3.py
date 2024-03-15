# ch3. 字典和集合
def test():
    # list
    # tuple
    # array

    # dict
    d = dict(one=1, two=2, three=3)
    print(d," ", d["one"])


    d3 = dict([('one',2), ('three',3), ('two', 1)])
    print(d3)

    d1 = dict(zip(['one_key', 'two_key', 'three_key'],[1, 2,3]))
    print(d1)

    d2 = {'one':1, 'two':2, 'three':3}

    d4 = ({'one':3, 'two':2, 'three': 1})
    print(d4)
    print(d4['two'])
    print(d4.keys())
    print(d4.values())
    print(d4.items())
    print(len(d4))

    # set
    l = ["what", "is", "your", "name"]
    s = set(l)
    print(s)
    l1 = list(s)
    print(l1)
    b = ["name"]
    a = set(l) - set(b)
    print(a)
    print(len(a))
    for x in a:
        print(x)

    a = frozenset(range(10))
    print(a)
    b = {1, 2, 3, 4, 5, 6, 7}
    print(b)
    print(type(b))
    b = [1, 2, 3]
    print(type(b))
    b = (1, 2, 3)
    print(type(b))



if __name__ == "__main__":
    test()
