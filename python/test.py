import time

def add(a, b):
    return a + b

def sum(a, b, c):
    d = 0
    if a < 0:
        d -= a
    else:
        d += a
    if b < 0:
        d -=  b
    else:
        d += b
    if c < 0:
        d -= c
    else:
        d += c

    return d


if __name__ == "__main__":
    c = add(2, 4)
    c = sum(1, 2, 3)
    print(c)
    c = sum(1, -2, -3)
    print(c)

    s = ["a", "b", "c", "d", "e"]
    for x in s:
        print(x)

    s.append("f")
    s[0]= 'g'
    s[2]= 'h'
    s.insert(6, 'q')
    print([x for x in s])
    print([y for y in range(10)])

