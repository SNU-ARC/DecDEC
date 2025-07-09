import math
import torch
import torch.nn as nn
import numpy as np

class Packer:
    def __init__(self):
        self.s = torch.from_numpy(np.array([1, 2, 4, 8, 16, 32, 64, 128])).view(
            [-1, 1])
        if torch.cuda.is_available():
            self.s = self.s.cuda()
        self.w_pool = {}

    def __get_weight(self, shape, dtype):
        key = np.prod(shape)
        if key not in self.w_pool.keys():
            self.w_pool[key] = torch.zeros(shape, dtype=dtype)
            if torch.cuda.is_available():
                self.w_pool[key] = self.w_pool[key].cuda()
        return self.w_pool[key].reshape(shape)

    def pack(self, b):
        shape = b.shape
        p_b = b
        if torch.cuda.is_available():
            p_b = p_b.cuda()
        p_b = (p_b + 1) / 2  # (-1., +1.) -> (0, 1)
        p_b = torch.reshape(p_b, [8, -1]).type(torch.uint8)
        p_b = p_b * self.s
        p_b = p_b.sum(0)
        p_b = p_b.type(torch.uint8)
        return p_b, shape

    def unpack(self, pb, shape, dtype=torch.float16):
        b = self.__get_weight(shape, dtype).view([8, -1])
        for i in range(8):
            b[i] = (pb & 1)  # (pB%2)
            pb = pb >> 1  # //2
        b = b * 2 - 1
        b = b.reshape(shape)
        return b

PACKER = Packer()
def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width
def convert_bcq_format(scale, zero, quant_data, qbits, do_packing=False, in_ch_wise=False):
    global PACKER

    zero   = scale * zero
    upack  = torch.Tensor([[2**i for i in range(qbits)]]).to(torch.device('cuda:0'))
    scale  = scale / 2.0
    scale  = torch.matmul(scale, upack)

    offset = scale.sum(-1).unsqueeze(-1) - zero
    offset= offset.reshape(offset.shape[0],-1)
    binary = torch.zeros(list(quant_data.shape) + [qbits],dtype =torch.int64)
    binary_shape = binary.shape
    
    quant_data = quant_data.to(torch.int)
    for i in range(qbits):
        binary[:, :, i] = ((quant_data >> i) & 1) # O I B

    N = binary.shape[0] #output

    scale_ = scale.permute(1,2,0).contiguous() # G B O
    binary_ = binary.permute(0,2,1).reshape(-1,32).contiguous().to(torch.int64).to(torch.device('cuda'))
    offset_ = offset.permute(1,0).contiguous() # G O

    matrix_132 = torch.tensor([1<<i for i in range(32)],dtype=torch.int64,device= "cuda")

    bW_ = binary_*matrix_132

    bW__ = torch.sum(bW_, dim=1, keepdim=True)
    bW__ = bW__.to(torch.int32)
    bW___ = bW__.reshape(N,qbits,-1).permute(2,1,0).contiguous()

    return scale_, bW___, binary_shape, offset_

def pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        #if w_bit not in [4]:
        #    raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8
        self.interleave = 4
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        #assert out_features % (32 // self.w_bit) == 0
        pack_num = 32 // 4
        int16_pack_num = 16 // 4

        assert out_features % (self.interleave) == 0
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    out_features,
                    in_features // group_size,
                    group_size
                ),
                dtype=torch.int32,
                device=dev,
            ),
        )

        self.register_buffer(
            "scales",
            torch.zeros(
                (
                    out_features,
                    in_features // group_size,
                    1
                ),
                dtype=torch.float32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scaled_zeros",
            torch.zeros(
                (
                    out_features,
                    in_features // group_size,
                    1
                ),
                dtype=torch.float32,
                device=dev,
            ),
        )          
        self.register_buffer(
            "q_bias",
            torch.zeros(
                (
                    in_features//128,
                    out_features
                ),
                dtype=torch.float32,
                device=dev,
            ),
        )
        self.register_buffer(
            "alpha",
            torch.zeros(
                (
                    in_features//128,
                    self.w_bit,
                    out_features
                ),
                dtype=torch.float32,
                device=dev,
            ),
        )
        self.register_buffer(
            "binary",
            torch.zeros(
                (
                    in_features//32,
                    self.w_bit,
                    out_features
                ),
                dtype=torch.int32,
                device=dev,
            ),
        )    
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization

        assert scales is not None and zeros is not None

        awq_linear.scaled_zeros = zeros.contiguous().reshape([zeros.shape[0], zeros.shape[1], 1])
        #print("1@ ",scales.size()," ",zeros.size())
        
        scale_zeros = zeros * scales

        pack_num = 32 // 4
        qscales = torch.zeros(
            (
                scales.shape[0],
                scales.shape[1],
            ),
            dtype=torch.float16,
            device=scales.device,
        )
        #print("3@",qscales.shape)
        qscales[:, : scales.shape[1]] = scales
        # awq_linear.scales = scales.clone().half()
        awq_linear.scales = qscales.contiguous().reshape([qscales.shape[0], qscales.shape[1], 1])
        awq_linear.scales = awq_linear.scales.to(dtype=torch.float32)
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        # intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32).reshape([intweight.shape[0], -1])
        awq_linear.qweight = intweight

        zeros = zeros.to(dtype=torch.int32)
        scaled_zeros = torch.zeros_like(qscales)
        # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
        scaled_zeros[:, : scales.shape[1]] = -(
            qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
        )

        awq_linear.scaled_zeros=awq_linear.scaled_zeros.to(dtype=torch.float32)
        zeros = zeros.reshape([zeros.shape[0],-1,1]).to(dtype=torch.float32)
        scales = scales.reshape([zeros.shape[0],-1,1]).to(dtype=torch.float32)
        #print(awq_linear.qweight)
        
        alpha, binary, binary_shape, offset = convert_bcq_format(
            scales, zeros, intweight, qbits=w_bit,
            do_packing=True, in_ch_wise=False)

        awq_linear.binary = binary
        awq_linear.alpha = alpha
        awq_linear.q_bias = offset

        return awq_linear
