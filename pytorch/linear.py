import torch
x = torch.randn(1, 20)
m= torch.nn.Linear(20, 30)
output = m(x)

print("m.weight.shape: ", m.weight.shape)
print("x.shape: ", x.shape)
print("output.shape", output.shape)

ans = torch.mm(x, m.weight.t()) + m.bias
print("ans.shape: ", ans.shape)
print(torch.equal(ans, output))
