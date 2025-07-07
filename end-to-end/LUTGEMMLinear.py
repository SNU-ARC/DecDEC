import torch
from DECLinear import DECLinear

from plugin import *

class LUTGEMMLinear(DECLinear):
    def __init__(self, in_features, out_features, bitwidth, group_size, bias=False, dtype=torch.half):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bitwidth = bitwidth
        self.group_size = group_size
        self.dtype = dtype

        self.register_buffer(
            'qweight',
            torch.empty((in_features//32, bitwidth, out_features), dtype=torch.int32, device='cuda')
        )

        self.register_buffer(
            'alpha',
            torch.empty((in_features // group_size, bitwidth, out_features), dtype=self.dtype, device='cuda')
        )
        
        self.register_buffer(
            'q_bias',
            torch.empty((in_features // group_size, out_features), dtype=self.dtype, device='cuda')
        )
       
        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=self.dtype, device='cuda')
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, out_features), dtype=self.dtype, device='cuda')

    def gemm(self, x):
        """  
        Dequantizes the output from a LU GEMM with reduced memory usage using FP16.
        """

        binary = self.qweight
        alpha = self.alpha
        q_bias = self.q_bias
        bit_width = self.bitwidth
        group_size = self.group_size

        device = binary.device                                                                        
                                                                                                      
        # Get shapes
        G_bin, B, O = binary.shape  # binary: [input_size // 32, bit_width, output_size]
        input_size = G_bin * 32                                                                       
        num_groups = input_size // group_size                                                         
        
        # Unpack bits from binary tensor                                                              
        shifts = torch.arange(32, dtype=torch.int32, device=device)                                   
        binary_expanded = binary.unsqueeze(-1)  # [G_bin, B, O, 1]                                    
        bits = ((binary_expanded >> shifts) & 1).byte()                                               
        bits = bits.permute(0, 3, 1, 2).reshape(input_size, B, O)                                     
            
        # In-place conversion of bits to delta, now using half precision
        delta = bits.to(torch.half).mul_(2).sub_(1)  # [input_size, bit_width, output_size], FP16     
                                                                                                      
        # Compute group indices without extra tensors                                                 
        group_indices = torch.div(                                                                    
            torch.arange(input_size, device=device), group_size, rounding_mode='floor'                
        ).long()  # [input_size]
                                                                                                      
        # Gather alpha and q_bias values per input dimension (already in FP16)                        
        alpha_values = alpha[group_indices, :, :]  # [input_size, bit_width, output_size], FP16       
        q_bias_values = q_bias[group_indices, :]  # [input_size, output_size], FP16                   
                                                                                                      
        # In-place multiplication and summing, keeping everything in FP16
        delta.mul_(alpha_values)                                                                      
        delta_alpha_sum = delta.sum(dim=1)  # [input_size, output_size], FP16                         
            
        # In-place addition of q_bias and transpose to avoid extra memory allocation                  
        return x @ (delta_alpha_sum + q_bias_values)


    def forward(self, x, **kwargs):
        assert(x.shape[0] == 1)

        if x.shape[1] > 1:
            output = self.gemm(x)
            if self.bias is not None:
                output += self.bias
            return output

        # clear the output
        self.output.zero_()

        dec_lutgemm(self.dec_config, x, self.output, self.qweight, self.alpha, self.q_bias, self.bitwidth, self.group_size)
        
        if self.bias is not None:
            self.output += self.bias

        return self.output
