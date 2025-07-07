# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config

import nvtx

from transformers import AutoTokenizer
from APLinear import APLinear
from LUTGEMMLinear import LUTGEMMLinear

import warnings

warnings.filterwarnings(
    "ignore", 
    category=FutureWarning
)

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
#torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits=logits.float()
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs): 
    new_tokens, new_probs = [], []
    with nvtx.annotate("Decoding tokens", color='orange'):
        for i in range(num_new_tokens):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                next_token, next_prob = decode_one_token(
                    model, cur_token, input_pos, **sampling_kwargs
                )
                
                # Update position and tokens
                input_pos += 1
                new_tokens.append(next_token.clone())
                callback(new_tokens[-1])
                new_probs.append(next_prob.clone())
                cur_token = next_token.clone()
        torch.cuda.synchronize()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:

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
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device, dtype=torch.int32)

    if T != 1:
        next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
        seq[:, T] = next_token.squeeze()
        input_pos = torch.tensor([T], device=device, dtype=torch.int).view(1)
        generated_tokens, new_probs = decode_n_tokens(model, next_token.view(batch_size, -1),
                          input_pos, max_new_tokens-1, callback=callback, **sampling_kwargs)
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    else:
        generated_tokens, new_probs = decode_n_tokens(model, prompt.view(batch_size, -1),
                              input_pos, max_new_tokens, callback=callback, **sampling_kwargs)
        seq[:, 1:] = torch.cat(generated_tokens, dim=-1)
        
    return seq

def encode_tokens(tokenizer, string, device=default_device):
    tokens = tokenizer.encode(string)
    return torch.tensor(tokens, dtype=torch.int, device=device)

def encode_bos(tokenizer, device=default_device):
    return torch.tensor([tokenizer.bos_token_id], dtype=torch.int, device=device)

def load_model(model_name, device, backend,  
                bitwidth, n_tbs, k_chunks, 
                checkpoint_path, dtype=None):
    use_cuda = 'cuda' in device

    model_path = os.path.join(checkpoint_path, "converted_pytorch_model.bin")
    dec_path = os.path.join(checkpoint_path, "cheatsheet.bin")

    bitwidth_map = None
    # Mixed precision, requires bitwidth map
    if not isinstance(bitwidth, int):
        with open(os.path.join(checkpoint_path, "bitwidth_map.json"), "r") as f:
            bitwidth_map = json.load(f)
            # convert layer index keys to int
            bitwidth_map = {int(k): v for k, v in bitwidth_map.items()}

    linear_kwargs = {"bitwidth": bitwidth}

    match backend:
        case "ap":
            linear_class = APLinear
        case "lutgemm":
            linear_class = LUTGEMMLinear
            linear_kwargs['group_size'] = 128

    print("Building model ...", flush=True)
    model = Transformer.from_name(
        name=model_name, dtype=dtype,
        linear_class=linear_class,
        linear_kwargs=linear_kwargs,
        bitwidth_map=bitwidth_map,
    )

    print("Loading weights ...", flush=True)
    checkpoint = torch.load(model_path, mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True, strict=True)

    print("Dispatching model to device ...", flush=True)
    model=model.to(device=device, dtype=dtype)

    # load & set up DEC
    print("Loading residuals ...", flush=True)
    buffer_size = 32 * 1024
    dec_data = torch.load(dec_path, weights_only=True)
    model.load_dec_data(dec_data)
    model.create_dec_context(n_tbs, buffer_size)
    model.set_dec_config(k_chunks)

    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(model_path))
   
    print("Model loaded.", flush=True)
    return model.eval(), tokenizer

def main(
    prompt, num_samples, max_new_tokens, top_k,
    temperature, compile, 
    device, model_name, backend, bitwidth,
    checkpoint_path, dtype, n_tbs, k_chunks,
    print_result,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    
    print(f"Using device={device}")
    if (dtype == "float16"): 
        dtype = torch.float16
    elif (dtype == "bfloat16"): 
        dtype = torch.bfloat16
    elif (dtype == "float32"): 
        dtype = torch.float32

    t0 = time.time()
    model, tokenizer = load_model(model_name, device, backend,
                                    bitwidth, n_tbs, k_chunks, 
                                    checkpoint_path, dtype)

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds", flush=True)

    # encode prompt
    if prompt != None:
        encoded = encode_tokens(tokenizer, prompt, device=device)
    else:
        encoded = encode_bos(tokenizer, device=device)
    prompt_length = encoded.size(-1)

    torch.manual_seed(1234)

    if compile:
        global decode_one_token
        mode = 'max-autotune-no-cudagraphs' if compile == 1 else 'max-autotune'
        decode_one_token = torch.compile(decode_one_token, mode=mode, fullgraph=True, dynamic=False)
        
    aggregate_metrics = {
        'tokens_per_sec': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        callback = lambda x : x
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        import contextlib
        prof = contextlib.nullcontext()
        with prof:
            y = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds", flush=True)
            continue

        torch.cuda.synchronize()
        time_elapsed = time.perf_counter() - t0

        if print_result:
            print(tokenizer.decode(y[0].tolist()))
        
        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / time_elapsed
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        print(f"Time for inference {i + 1}: {time_elapsed:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print(flush=True)
    print("==========")

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=int_or_str, default=None, help="Input prompt. If it is not given, a single BOS token is fed as input prompt.")
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling.')
    parser.add_argument('--compile', type=int, default=2, help='Whether to compile the model')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model', choices=['Meta-Llama-3-8B-Instruct', 'Phi-3-medium-4k-instruct'])
    parser.add_argument('--bitwidth', type=str, default=None, help='Quantization bitwidth', choices=['3', '4', '3.5'])
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--dtype', type=str, default="float16", help='Data type for GEMV computation', choices=["float16", "float32", "bfloat16"])
    parser.add_argument('--backend', type=str, default=None, help='Quantized GEMV kernel to use', choices=["ap", "lutgemm"])
    parser.add_argument('--n_tbs', nargs='+', type=int, default=None, help='n_tb, provide 2 for mixed precision')
    parser.add_argument(
                '--k_chunks',
                nargs='+',  # '+' means one or more arguments
                type=int,   # Type conversion to integer
                help='A list of k_chunk [qkv, o, gu, d]. For mixed precision, supply 8 values, 4 for each bitwidth.',
                )
    parser.add_argument('--print_result', action='store_true')

    args = parser.parse_args()

    # Convert bitwidth to int or float
    if '.' in args.bitwidth:
        bitwidth = float(args.bitwidth)
    else:
        bitwidth = int(args.bitwidth)

    main(
        args.prompt, args.num_samples, args.max_new_tokens, args.top_k,
        args.temperature, args.compile, 
        args.device, args.model_name, args.backend, bitwidth, 
        args.checkpoint_path, args.dtype, args.n_tbs, args.k_chunks, 
        args.print_result, 
    )

