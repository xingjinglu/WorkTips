import torch

x = torch.rand(3, 2)
#indices = [ [0, 1], [1, 2],]
indices = torch.tensor([[0,1],[1,2],])

print(x.shape)
print(x)
print(indices.shape)
print(indices)

z = torch.gather(x, 0, indices)
print(z.shape)
print(z)
