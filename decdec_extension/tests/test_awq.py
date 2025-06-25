import torch
import decdec_ext

in_ch = 4096
out_ch = 1024
bitwidth = 4
group_size = 128

num_groups = in_ch // group_size
torch.manual_seed(0)

# Randomize alpha, qweight, and q_bias
alpha_fp32 = torch.rand((num_groups, bitwidth, out_ch), dtype=torch.float32).cuda()
qweight = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (in_ch // 32, bitwidth, out_ch), dtype=torch.int32).cuda()
q_bias_fp32 = torch.rand((num_groups, out_ch), dtype=torch.float32).cuda()

# Initialize input in base type
x_fp32 = torch.rand((1, 1, in_ch), dtype=torch.float32).cuda()

# Define the data types to compare
dtypes = [torch.float32, torch.float16, torch.bfloat16]

# Loop through each data type
for dtype in dtypes:
    x = x_fp32.to(dtype)
    alpha = alpha_fp32.to(dtype)
    q_bias = q_bias_fp32.to(dtype)
    output = torch.zeros((1, 1, out_ch), dtype=dtype).cuda()

    decdec_ext.lutgemm_gemv(x, qweight, alpha, q_bias, output, bitwidth, group_size)

    # Print the first 5 elements of the output
    print(f"{str(dtype):<15}: " + ', '.join([f"{output[0, 0, i]:.4f}" for i in range(5)]))
