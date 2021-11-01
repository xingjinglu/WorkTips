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

# resnet50 shape->gather->unsqueeze
x = torch.tensor([1, 2188, 1, 1])
print("input: ", x.shape, x)
indices = torch.tensor(0)
#indices = torch.tensor([0], dtype=torch.int64)
print("indices: ", indices.shape, indices)
z = torch.gather(x, 0, indices)
print("gather: ", z.shape, z)
y = torch.unsqueeze(z, 0)
print("unsqueeze: ", y.shape, y)
