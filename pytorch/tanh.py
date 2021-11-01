import torch 

a = torch.randn(2)
a[0] = 574.17
a[1] = 574.17
b = torch.tanh(a)
print(b)
