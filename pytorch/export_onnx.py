import torch
import io
import numpy as np
import torch.onnx
import torch.nn as nn

class TestCholesky(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(TestCholesky, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

        #self._initialize_weights()

    
    def forward(self, x):
        #x = self.relu(x)
        x = torch.cholesky(x)
        return x


torch_model = TestCholesky(upscale_factor=3)

x = torch.randn(3, 3)

#torch_model.eval()

#torch_out = torch_model(x)

a = torch.randn(3, 3, dtype=torch.float32)
l = torch.linalg.cholesky(a)
print(l)

'''
torch.onnx.export(torch_model,
        x,
        "test_cholesky.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'])
'''



        



