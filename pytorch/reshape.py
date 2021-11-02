import torch
import numpy as np

a = torch.arange(16.0)
print(a)
print(torch.reshape(a, (-1,)))
print(torch.reshape(a, (-1,4)))
