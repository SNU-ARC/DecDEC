import torch
import decdec_ext

#in_ch = 4096
#out_ch = 14336
in_ch = 1024 * 32
out_ch = 1024 * 32
bitwidth = 3

torch.manual_seed(0)

# Initialize x, lut, and qweight in their base types
x_fp32 = torch.rand((1, 1, in_ch), dtype=torch.float32).cuda()
qweight_int = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (bitwidth, out_ch, in_ch // 32), dtype=torch.int32).cuda()
lut_fp32 = torch.rand(out_ch, 2 ** bitwidth, dtype=torch.float32).cuda()

# Define the data types to compare
#dtypes = [torch.float16, torch.bfloat16]
dtypes = [torch.float16]

# Loop through each data type
for dtype in dtypes:
    x = x_fp32.to(dtype)
    lut = lut_fp32.to(dtype)
    output = torch.zeros((1, 1, out_ch), dtype=dtype).cuda()

    decdec_ext.anyprec_gemv(x, output, qweight_int, lut, bitwidth)

    dequantized = decdec_ext.anyprec_dequant(qweight_int, lut, bitwidth)

    dequantized_results = x @ dequantized.t()

    # print the first 10 elements of the output
    print(f"{str(dtype):<15}: " + ', '.join([f"{output[0, 0, i]:.4f}" for i in range(5)]))
    # print the first 10 elements of the dequantized_results
    print(f"{str(dtype) + ' deq':<15}: " + ', '.join([f"{dequantized_results[0, 0, i]:.4f}" for i in range(5)]))
