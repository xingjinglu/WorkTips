import torch

t = torch.tensor([[[1, 2],
                   [3, 4]],
                   [[5, 6],
                    [7, 8]]])

print(t.shape)
print(t)
t1 = torch.flatten(t)
print(t1)
t2 = torch.flatten(t, start_dim=2)
print(t2.shape)
print(t2)



