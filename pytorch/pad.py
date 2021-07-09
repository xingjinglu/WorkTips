import torch
input = torch.empty(2, 2, 4, 2)
p1d = (0 , 0, 0, 0)
out = torch.nn.functional.pad(input, p1d, "constant", 0)
print(out)
