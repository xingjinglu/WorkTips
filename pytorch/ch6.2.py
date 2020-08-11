import torch
X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)

O = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)

print(O.type())
print(O)



