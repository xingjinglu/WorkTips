import torch.nn as nn
import torch

class TestCholesky(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(TestCholesky, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

        #self._initialize_weights()


    def forward(self, x):
        #x = self.relu(x)
        #x = x @ x.mT + 1e-3
        x = torch.cholesky(x)
        return x


torch_model = TestCholesky(upscale_factor=3)

#x = torch.randn(3, 3)
x = torch.tensor([[ 2.3999,  2.2497, -1.7825],
        [ 2.2497,  3.1641, -3.4624],
        [-1.7825, -3.4624,  4.5418]])

#torch_model.eval()

#torch_out = torch_model(x)

a = torch.randn(3, 3, dtype=torch.float32)
a = a @ a.mT + 1e-3
print(a)
l = torch.linalg.cholesky(a)
print(l)


#'''
torch.onnx.export(torch_model,
        x,
        "test_cholesky.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'])
#        '''
