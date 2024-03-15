import numpy as np
import torch


def test():
    a = np.ndarray(shape=(3, 4), dtype=float)
    print(a)
    print(type(a))
    print("a_addr: ", id(a))
    #a[:,:] = torch.tensor(a)
    #a = torch.tensor(a)
    a[:,:] = torch.tensor(a)
    print(type(a))
    print("a_addr: ", id(a))

if __name__ == "__main__":
    test()
