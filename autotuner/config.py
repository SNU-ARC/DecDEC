model_configurations = {
    'Phi-3-medium-4k-instruct': {
        'qkv': (5120, 7680),
        'o': (5120, 5120),
        'gu': (5120, 35840),
        'd': (17920, 5120),
    },
    # 'Meta-Llama-3-8B-Instruct': {
    #     'qkv': (4096, 6144),
    #     'o': (4096, 4096),
    #     'gu': (4096, 28672),
    #     'd': (14336, 4096),
    # },
    'Meta-Llama-3-70B-Instruct': {
        'qkv': (8192, 10240),
        'o': (8192, 8192),
        'gu': (8192, 57344),
        'd': (28672, 8192),
    },
}

_halve_layers = {  # Whether to halve the number of layers in the model to prevent OOM
    #('Phi-3-medium-4k-instruct', 'awq', 4)
}

def halve_layers(model_name, algo, bit_width):
    return (model_name, algo, bit_width) in _halve_layers

algos = ['awq', 'sqllm']
bitwidths = [3, 4]

fp_dtype = 'fp16'
nsys_path = "/usr/local/bin/nsys"
profiler_path = '/work/syphon/end-to-end/gpt-fast/profiler.py'
csv_output_path = 'timings.csv'
tokens_per_config = 10

dec_kernel_name = 'fused_dec_kernel'

def get_gemv_kernel_name(algo, bit_width):
    match algo:
        case 'sqllm':
            return f'matmul_kbit_32'
        case 'awq':
            return f'nqmv_bias'
        case _:
            raise ValueError(f"Unknown quantization algorithm: {algo}")

def get_output_path(prefix, model_name, fp_dtype, base_gemv, bit_width):
    return f"./nsys/{prefix}_{model_name}_{fp_dtype}_{base_gemv}_{bit_width}"
