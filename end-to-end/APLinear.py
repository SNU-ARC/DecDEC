import torch
from DECLinear import DECLinear

from plugin import *
import decdec_ext

class APLinear(DECLinear):
    def __init__(self, in_features, out_features, bitwidth, bias=False, dtype=torch.half):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bitwidth = bitwidth
        self.dtype = dtype

        self.register_buffer(
            'qweight',
            torch.empty((bitwidth, out_features,  in_features // 32), dtype=torch.int32, device='cuda')
        )

        self.register_buffer(
            'lut',
            torch.empty((out_features, 2 ** bitwidth), dtype=self.dtype, device='cuda')
        )
       
        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=self.dtype, device='cuda')
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, self.out_features), dtype=self.dtype, device='cuda')
    
    def gemm(self, x):
        weight = decdec_ext.anyprec_dequant(self.qweight, self.lut, self.bitwidth)
        output = torch.matmul(x, weight.T)
        return output

    def forward(self, x, **kwargs):
        assert(x.shape[0] == 1)
        
        if x.shape[1] > 1:
            output = self.gemm(x)
            if self.bias is not None:
                output += self.bias
            return output

        # clear the output
        self.output.zero_()

        dec_anyprec(self.dec_config, x, self.output, self.qweight, self.lut, self.bitwidth)

        if self.bias is not None:
            self.output += self.bias

        return self.output
