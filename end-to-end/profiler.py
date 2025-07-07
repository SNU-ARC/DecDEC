import torch
import time
import generate
from itertools import product
from tqdm import tqdm
import nvtx
from model import Transformer


def get_gemv_kernel_name(algo, bit_width):
    match algo:
        case 'sqllm':
            return f'matmul_kbit_32'
        case 'awq':
            return f'nqmv_bias'
        case _:
            raise ValueError(f"Unknown quantization algorithm: {algo}")


def generate_across_configs(gemv_kernel_name, model, tokenizer, tokens_per_config, min_n_tb, max_n_tb, max_k_chunk):
    """Generates text for each configuration of n_tb and k_chunk and returns timing and decoded outputs."""
    trace = []

    # If CUDA graphs are enabled, the kernel parameters are fixed and cannot be changed.
    # By using max-autotune-no-cudagraphs, we compile the kernels but don't use CUDA graphs.
    # The compilation happens on the first invocation of the function, which will be during the
    # comile run below. On subsequent invocations, the compiled kernels are used, and we can
    # safely create CUDA graphs per configuration without recompiling the kernels.
    # This approach is effective because running torch.compile for each configuration is way too slow.
    generate.decode_one_token = torch.compile(
        generate.decode_one_token, 
        mode="max-autotune-no-cudagraphs", 
        fullgraph=True, 
        dynamic=False
    )

    num_layers = model.config.n_layer
    kernel_calls_per_module_per_config = num_layers * tokens_per_config

    n_tb_list = range(min_n_tb, max_n_tb + 1)
    k_chunk_list = range(0, max_k_chunk + 1)

    # Perform model compilation with the first invocation
    print("Compiling the model...", flush=True)
    with nvtx.annotate(message="Model Compilation", color="blue"):
        # Perform a single generation to compile the model
        # use_graph=False so that torch.compile and CUDA graphs don't interfere
        perform_generation(model, tokenizer, tokens_per_config, n_tb=1, k_chunk=1, use_graph=False)
        record_trace(trace, 'compile', kernel_calls_per_module_per_config, gemv_kernel_name, k_chunk=1)
    print("Model compiled!", flush=True)

    # Warmup Run
    with nvtx.annotate(message="Warmup Region", color="yellow"):
        perform_generation(model, tokenizer, tokens_per_config, n_tb=1, k_chunk=1, use_graph=True)
        record_trace(trace, 'warmup', kernel_calls_per_module_per_config, gemv_kernel_name, k_chunk=1)

    concurrent_idx = 0
    # Main Profiling Loop
    with tqdm(total=len(n_tb_list) * len(k_chunk_list)) as pbar:
        for n_tb in n_tb_list:
            with nvtx.annotate(message=f"n_tb: {n_tb}", color="purple"):
                for k_chunk in k_chunk_list:
                    with nvtx.annotate(message=f"k_chunk: {k_chunk}", color="green"):
                        perform_generation(model, tokenizer, tokens_per_config, n_tb, k_chunk, use_graph=True)
                        concurrent_idx = record_trace(
                            trace, 'concurrent', kernel_calls_per_module_per_config, gemv_kernel_name, k_chunk, concurrent_idx, 
                        )
                    pbar.update(1)

    print("", flush=True, end="")  # To flush the end of the progress bar

    return trace

def perform_generation(model, tokenizer, tokens_per_config, n_tb, k_chunk, use_graph=True):
    """Performs the model generation and synchronization."""
    model.update_dec_context(n_tb)
    model.update_dec_config([k_chunk] * 4)

    seq = _generate(
        model,
        generate.encode_bos(tokenizer, 'cuda'),
        tokens_per_config,
        use_graph,
    )

    #print(tokenizer.decode(seq[0].tolist()))

@torch.no_grad()
def _generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    use_graph: bool = True,
) -> torch.Tensor:
    batch_size = 1

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device, dtype=torch.int32)


    generated_tokens, new_probs = generate.decode_n_tokens(model, prompt.view(batch_size, -1),
                        input_pos, max_new_tokens, use_graph, temperature=0.0, top_k=200)
    

    seq[:, 1:] = torch.cat(generated_tokens, dim=-1)

    return seq



def record_trace(trace, call_type, kernel_calls_per_module_per_config, gemv_kernel_name, k_chunk, concurrent_idx=None):
    """Records the trace of kernel calls."""
    for _ in range(kernel_calls_per_module_per_config):
        for module_name in ('qkv', 'o', 'gu', 'd'):
            match call_type:
                case 'warmup' | 'compile':
                    trace.append(f'{call_type} {gemv_kernel_name} {module_name}')
                    trace.append(f'{call_type} fused_dec_kernel 1')
                case 'concurrent':
                    if k_chunk != 0:
                        assert concurrent_idx is not None
                        trace.append(f'concurrent {concurrent_idx} fused_dec_kernel {k_chunk}')
                        trace.append(f'concurrent {concurrent_idx} {gemv_kernel_name} {module_name}')
                        concurrent_idx += 1
                    else:
                        trace.append(f'standalone {gemv_kernel_name} {module_name}')
                case _:
                    raise ValueError(f"Unknown call type: {call_type}")
    return concurrent_idx


def main(
    dtype,
    quant_algo,
    bitwidth,
    model_name,
    halve_layers,
    checkpoint_path,
    min_n_tb,
    max_n_tb,
    max_k_chunk,
    tokens_per_config,
    trace_file,
):
    print("==================================================================")
    print(f"Profiling model: {model_name}")
    print(f"Quantization algorithm: {quant_algo}")
    print(f"Bitwidth: {bitwidth}")
    print(f"Threadblocks: {min_n_tb} - {max_n_tb}")
    print(f"Max k_chunk: {max_k_chunk} per 1024")
    print(f"Tokens per config: {tokens_per_config}")
    print("==================================================================", flush=True)
    # Load model and tokenizer once, outside the compiled function

    if quant_algo == 'sqllm':
        backend = 'ap'
    elif quant_algo == 'awq':
        backend = 'lutgemm'
    else:
        raise ValueError(f"Unknown quantization algorithm: {quant_algo}")

    with nvtx.annotate(message="Loading Model", color=0xe706f9):
        model, tokenizer = generate.load_model(
            model_name, 'cuda', backend, bitwidth, n_tbs=1, k_chunks=[1, 1, 1, 1], dtype=dtype,
             checkpoint_path=checkpoint_path,halve_layers=halve_layers,
        )
    # Run the generate_across_configs function and get results
    gemv_kernel_name = get_gemv_kernel_name(quant_algo, bitwidth)

    trace = generate_across_configs(
        gemv_kernel_name, model, tokenizer, tokens_per_config, min_n_tb, max_n_tb, max_k_chunk
    )

    print(f"Writing trace to {trace_file}", flush=True)
    with open(trace_file, 'w') as f:
        f.write('\n'.join(trace))
    print("Done!", flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmarking script for model generation.')

    parser.add_argument('--dtype', type=str, default="fp16", help='dtype',
                        choices=["fp16", "fp32", "bf16"])
    parser.add_argument('--quant_algo', type=str, help='Quantization algorithm',
                        choices=["sqllm", "awq"], default="sqllm")
    parser.add_argument('--bitwidth', type=int, help='Bitwidth', choices=[3, 4], default=3)
    parser.add_argument('--model_name', type=str, help='Model name',
                        default='Meta-Llama-3-8B-Instruct')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--min_n_tb', type=int, help='Min number of threadblocks', default=8)
    parser.add_argument('--max_n_tb', type=int, help='Max number of threadblocks', default=24)
    parser.add_argument('--max_k_chunk', type=int, help='Max k_chunk', default=8)
    parser.add_argument('--tokens_per_config', type=int, help='Number of tokens per config', default=10)
    parser.add_argument('--trace_file', type=str, help='File to write trace to', default='trace.txt')
    parser.add_argument('--halve_layers', action='store_true', help='Halve the number of layers to profile, useful to avoid OOM')

    args = parser.parse_args()

    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }

    main(
        dtype=dtype_map[args.dtype],
        quant_algo=args.quant_algo,
        bitwidth=args.bitwidth,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        halve_layers=args.halve_layers,
        min_n_tb=args.min_n_tb,
        max_n_tb=args.max_n_tb,
        max_k_chunk=args.max_k_chunk,
        tokens_per_config=args.tokens_per_config,
        trace_file=args.trace_file,
    )
