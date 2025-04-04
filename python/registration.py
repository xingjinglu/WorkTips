registry = []

def register(func):
    print('running register {}'.format(func))
    registry.append(func)
    return func

@register
def f1():
    print('running f1()')

@register
def f2():
    print('running f1()')

@register
def f3():
    print('running f1()')

def main():
    print('running main()')
    print('registry->', registry)
    f1()
    f2()
    f3()

if __name__ == "__main__":
    main()
