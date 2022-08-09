import torch
import io
import numpy as np
import torch.onnx
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(TestModel, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

        #self._initialize_weights()


    def forward(self, x):
        #x = self.relu(x)
        #x = torch.cholesky(x)
        x1 = torch.tril(x)
        return x1


torch_model = TestModel(upscale_factor=3)
a = torch.randn(3, 3)
b = torch_model(a)
print(b)

torch.onnx.export(torch_model,
        a,
        "tril.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'])

