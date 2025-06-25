import torch
import decdec_ext

# Set up dimensions and configuration
num_tb = 24
num_rows = 14336
num_cols = 4096
k_chunk = 20
buffer_size = num_rows

torch.manual_seed(0)

# Create buffers and tensors for the configuration
row_buffer = torch.zeros(buffer_size, dtype=torch.int32).cuda()
act_buffer = torch.zeros(buffer_size, dtype=torch.float32).cuda()  # Default dtype for initialization
q_residual = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (num_rows, num_cols // 8), dtype=torch.int32)

reordered_scales = torch.rand(num_cols, dtype=torch.float32)

x = torch.rand((1, 1, num_rows), dtype=torch.float32).cuda() * 2 - 1
thresholds = torch.sort(torch.abs(x.cpu()).view(-1), descending=True).values

# Define the data types to compare
dtypes = [torch.float32, torch.float16, torch.bfloat16]
#dtypes = [torch.float16, torch.bfloat16]
#dtypes = [torch.float16]

# Loop through each data type
for dtype in dtypes:
    # Convert act_buffer, reordered_scales, and thresholds to the current dtype
    act_buffer_typed = act_buffer.to(dtype)
    reordered_scales_typed = reordered_scales.to(dtype)
    thresholds_typed = thresholds.to(dtype)

    # Initialize dec_context and dec_config with the current dtype
    dec_context = decdec_ext.create_dec_context(num_tb, row_buffer, act_buffer_typed)
    dec_config = decdec_ext.create_dec_config(dec_context, k_chunk, q_residual, reordered_scales_typed, thresholds_typed)

    x_typed = x.to(dtype)
    # Input and output tensors in the current dtype
    output = torch.zeros((1, 1, num_cols), dtype=dtype).cuda()

    # Run dec function
    decdec_ext.dec(dec_config, x_typed, output)

    # Print the first 5 elements of the output
    print(f"{str(dtype):<15}: " + ', '.join([f"{output[0, 0, i]:.4f}" for i in range(5)]))
