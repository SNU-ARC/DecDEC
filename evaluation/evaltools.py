import torch
import contextlib
import sys
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoConfig
import numpy as np
import json
import lm_eval
import decdec_ext
import os
from accelerate import init_empty_weights
from datasets import load_dataset
import warnings
from itertools import product

DTYPE = torch.float16

def get_layers(model):
    for name, module in model.model.named_children():
        if isinstance(module, torch.nn.ModuleList):
            return name, module
    else:
        raise ValueError("Model layers not found")

def get_module_names(model):
    layers_name, layers = get_layers(model)
    first_layer = next(layers.children())
    # find all linear layers
    module_names = []
    for name, module in first_layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_names.append(name)
    return module_names

def logprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2():
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return "\n\n".join(testdata['text'])


def get_ptb():
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return "\n\n".join(valdata['sentence'])


def get_c4():
    raise NotImplementedError("Only C4-new has been refactored to use the new dataset API")


def get_ptb_new():
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    return " ".join(testdata['sentence'])


def get_ptb_new_sliced():
    raw_text = get_ptb_new()
    sliced = raw_text.replace('<unk>', '< u n k >')
    return sliced


def get_c4_new():
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation'
    )
    # The original datautils from the GPTQ paper had two filters:
    #   1. get only the first 1100 examples
    #   2. tokenize, then get the first seqlen * 256 tokens, where seqlen defaulted to 2048
    # This resulted in 524288 tokens, which in turn decode back into 2088532 characters.

    # However in my version, I am only returning the text, and leaving the tokenization to the caller.
    # Therefore, I replace the second filter of tokens into an equivalent filter of characters.
    return " ".join(valdata[:1100]['text'])[:2088528]


def get_loaders(name):
    if 'wikitext2' in name:
        return get_wikitext2()
    if 'ptb' in name:
        if 'new' in name:
            if 'sliced' in name:
                return get_ptb_new_sliced()
            else:
                return get_ptb_new()
        return get_ptb()
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new()
        return get_c4()

    raise ValueError(f"Unknown dataset {name}")


def _load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose):
    """ Load input tokens from cache if available, otherwise load from dataloader and save to cache. """
    logprint(verbose, "Loading test set...")

    raw_text = get_loaders(testcase_name)

    logprint(verbose, "Tokenizing test set...")

    input_tokens = tokenizer(raw_text, return_tensors='pt')
    return input_tokens

@torch.no_grad()
def evaluate_ppl(model, tokenizer, testcases, verbose=True, chunk_size=2048, tokenizer_type=None):
    """
    Args:
        model: model to evaluate
        tokenizer: tokenizer to use
        testcases: testcases names to evaluate on, passed on to dataloader.get_loaders
        verbose: whether to print progress
        chunk_size: the size of the chunks into which the test set is split
        tokenizer_type: set to llama, llama-2, or opt to use cached input tokens
                        for the corresponding test set

    Returns:
        A dictionary of perplexity scores, with keys being the testcases names and values being the perplexity scores.

    Note that the perplexity scores are calculated over non-overlapping chunks of the test set.
    """

    model.eval()

    results = {}
    for testcase_name in testcases:
        vprint(verbose, f"---------------------- {testcase_name} ----------------------")

        input_tokens = _load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose)

        logprint(verbose, "Calculating perplexity...")

        seq_len = input_tokens.input_ids.size(1)
        nsamples = seq_len // chunk_size  # floor(seq_len / chunk_size)

        neg_log_likelihoods = []
        for i in tqdm(range(nsamples), disable=not verbose):
            begin_loc = i * chunk_size

            input_ids = input_tokens.input_ids[:, begin_loc:begin_loc + chunk_size].to(model.device)

            # add BOS token for Gemma-7B
            # https://github.com/huggingface/transformers/issues/29250
            if 'gemma' in model.config.architectures[0].lower():
                # Mostly harmless to other models, but a slight drop in ppl is observed
                # Hence, we only add the BOS token for Gemma models for now
                input_ids[:, 0] = tokenizer.bos_token_id

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss.cpu()
                neg_log_likelihoods.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
        logprint(verbose, f"Perplexity: {ppl.item()}")

        results[f"{testcase_name}"] = ppl.item()

    return results

@torch.no_grad()
def run_lm_eval(tokenizer, model, tasks, verbose=True):
    """ Run lm-eval on the given model and tasks and return the results.

    Receives an already initialized hf model, and a list of task names.
    """
    model.eval()

    results = {}

    model_lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    eval_results = lm_eval.simple_evaluate(model=model_lm, tasks=tasks, batch_size=1)

    print(json.dumps(eval_results['results'], indent=4))

    return eval_results['results']

# Custom linear layer with an overridden forward method
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.int32_buffer = torch.zeros(in_features, device='cuda', dtype=torch.int32)
        self.half_buffer = torch.zeros(in_features, device='cuda', dtype=DTYPE)

    @property
    def k(self):
        return self.k_chunk * self.in_features // 1024

    def gemv(self, input):
        if self.algo == 'lutgemm':
            return self.lutgemm_gemv(input)
        elif self.algo == 'sqllm':
            return self.sqllm_gemv(input)
        elif self.algo == 'anyprec':
            return self.anyprec_gemv(input)
        elif self.algo == 'fake':
            return input @ self.weight.t()
        else:
            raise ValueError(f"Invalid algo {self.algo}")

    def lutgemm_gemv(self, input):
        output = torch.zeros(input.size(0), input.size(1), self.out_features, device=input.device, dtype=DTYPE)
        decdec_ext.lutgemm_gemv(
            input,
            output,
            self.binary,
            self.alpha,
            self.q_bias,
            self.bitwidth,
            128
        )
        return output

    def sqllm_gemv(self, input):
        output = torch.zeros(input.size(0), input.size(1), self.out_features, device=input.device, dtype=DTYPE)
        decdec_ext.sqllm_gemv(
            input,
            output,
            self.qweight,
            self.lut,
            self.bitwidth,
        )
        return output

    def anyprec_gemv(self, input):
        output = torch.zeros(input.size(0), input.size(1), self.out_features, device=input.device, dtype=DTYPE)
        lut = getattr(self, f'lut{self.bitwidth}')
        decdec_ext.anyprec_gemv(
            input,
            output,
            self.qweight,
            lut,
            self.bitwidth,
        )
        return output

    def gemm(self, input):
        if self.algo == 'lutgemm':
            output = self.lutgemm_gemm(input)
        elif self.algo == 'sqllm':
            output = self.sqllm_gemm(input)
        elif self.algo == 'anyprec':
            output = self.anyprec_gemm(input)
        elif self.algo == 'fake':
            output = input @ self.weight.t()
        else:
            raise ValueError(f"Invalid algo {self.algo}")
        return output

    def lutgemm_gemm(self, input):
        weights = lutgemm_dequant(self.binary, self.alpha, self.q_bias, self.bitwidth, 128)
        return input @ weights.t()

    def sqllm_gemm(self, input):
        weights = sqllm_dequant(self.qweight, self.lut, self.bitwidth)
        return input @ weights.t()

    def anyprec_gemm(self, input):
        lut = getattr(self, f'lut{self.bitwidth}')
        weights = anyprec_dequant(self.qweight, lut, self.bitwidth)
        return input @ weights.t()

    def forward_decdec(self, input):
        batch_size, seq_len, hidden_size = input.size()
        if seq_len != 1:
            # Prefill phase
            return self.gemm(input)
        else:
            # Decode phase
            output = self.gemv(input)
            decdec_ext.dec(self.dec_config, input, output)

            return output

    def forward_exact_decdec(self, input):
        batch_size, seq_len, hidden_size = input.size()
        if seq_len != 1:
            # Prefill phase
            return self.gemm(input)
        else:
            # Decode phase
            original_output = self.gemv(input)

            decdec_ext.dec(self.dec_config, input, torch.zeros_like(original_output))

            selected_indices = self.int32_buffer[:self.k]
            error = input[0][0][selected_indices] @ self.exact_residuals.t()[selected_indices]
            original_output[0][0] += error

            return original_output

    def forward_exact_decdec_ppl(self, input):
        batch_size, seq_len, hidden_size = input.size()
        assert batch_size == 1, "Batch size must be 1"
        # Decode phase
        original_output = self.gemm(input)

        if self.k_chunk == 0:
            return original_output

        temp_buffer = torch.zeros_like(original_output[0][0]).unsqueeze(0).unsqueeze(0)
        for seq_idx in range(seq_len):
            decdec_ext.dec(self.dec_config, input[0][seq_idx].unsqueeze(0).unsqueeze(0), temp_buffer)

            selected_indices = self.int32_buffer[:self.k]
            error = input[0][seq_idx][selected_indices] @ self.exact_residuals.t()[selected_indices]
            original_output[0][seq_idx] += error

        return original_output

    def forward_topk(self, input):
        batch_size, seq_len, hidden_size = input.size()
        if seq_len != 1:
            # Prefill phase
            return self.gemm(input)
        else:
            # Decode phase
            original_output = self.gemv(input)

            topk_indices = torch.topk(input[0][0].abs(), self.k, largest=True).indices.to(torch.int32)

            # copy topk_indices to the buffer
            self.int32_buffer[:self.k] = topk_indices

            decdec_ext.unfused_dec(self.dec_config, input, original_output)

            return original_output

    def forward_ppl(self, input):
        batch_size, seq_len, hidden_size = input.size()
        assert batch_size == 1, "Batch size must be 1"
        # Decode phase
        original_output = self.gemm(input)

        if self.k_chunk == 0:
            return original_output

        for seq_idx in range(seq_len):
            decdec_ext.dec(
                self.dec_config,
                input[0][seq_idx].unsqueeze(0).unsqueeze(0),
                original_output[0][seq_idx].unsqueeze(0).unsqueeze(0)
            )

        return original_output

    def forward_topk_check_ppl(self, input):
        batch_size, seq_len, hidden_size = input.size()
        assert batch_size == 1, "Batch size must be 1"
        # Decode phase
        original_output = self.gemm(input)

        if self.k_chunk == 0:
            return original_output

        # Call topk once for all sequence indices
        topk_indices_all = torch.topk(input.abs(), self.k, dim=2, largest=True).indices.to(torch.int32)

        # Move to CPU and convert to NumPy just once
        topk_indices_all_numpy = topk_indices_all.cpu().numpy()

        for seq_idx in range(seq_len):
            decdec_ext.dec(
                self.dec_config,
                input[0][seq_idx].unsqueeze(0).unsqueeze(0),
                original_output[0][seq_idx].unsqueeze(0).unsqueeze(0)
            )

            # Use precomputed topk indices from NumPy array
            topk_indices_set = set(topk_indices_all_numpy[0][seq_idx])

            # Convert the buffer to NumPy once
            selected_indices_set = set(self.int32_buffer[:self.k].cpu().numpy())

            # Check the size of the intersection
            intersection = topk_indices_set.intersection(selected_indices_set)
            ratio = len(intersection) / self.k
            if not hasattr(self, 'topk_intersection_ratio'):
                self.topk_intersection_ratio = []
            self.topk_intersection_ratio.append(ratio)

        return original_output

    def forward_topk_ppl(self, input):
        batch_size, seq_len, hidden_size = input.size()
        assert batch_size == 1, "Batch size must be 1"
        # Decode phase
        original_output = self.gemm(input)

        if self.k_chunk == 0:
            return original_output

        # Call topk once for all sequence indices
        topk_indices_all = torch.topk(input.abs(), self.k, dim=2, largest=True).indices.to(torch.int32)

        for seq_idx in range(seq_len):
            # Copy precomputed topk indices to the buffer
            self.int32_buffer[:self.k] = topk_indices_all[0, seq_idx]

            decdec_ext.unfused_dec(
                self.dec_config,
                input[0][seq_idx].unsqueeze(0).unsqueeze(0),
                original_output[0][seq_idx].unsqueeze(0).unsqueeze(0),
            )

        return original_output

    def forward_static(self, input):
        batch_size, seq_len, hidden_size = input.size()
        if seq_len != 1:
            # Prefill phase
            return self.gemm(input)
        else:
            # Decode phase
            original_output = self.gemv(input)

            if self.k_chunk == 0:
                return original_output
            
            # copy static indices to the buffer
            self.int32_buffer[:self.k] = self.static_topk[:self.k]

            decdec_ext.unfused_dec(
                self.dec_config,
                input,
                original_output,
            )

            return original_output


    def forward_static_ppl(self, input):
        batch_size, seq_len, hidden_size = input.size()
        assert batch_size == 1, "Batch size must be 1"
        # Decode phase
        original_output = self.gemm(input)

        if self.k_chunk == 0:
            return original_output

        # copy static indices to the buffer
        self.int32_buffer[:self.k] = self.static_topk[:self.k]

        for seq_idx in range(seq_len):
            decdec_ext.unfused_dec(
                self.dec_config,
                input[0][seq_idx].unsqueeze(0).unsqueeze(0),
                original_output[0][seq_idx].unsqueeze(0).unsqueeze(0),
            )

        return original_output


    def forward_random(self, input):
        batch_size, seq_len, hidden_size = input.size()
        if seq_len != 1:
            # Prefill phase
            return self.gemm(input)
        else:
            # Decode phase
            original_output = self.gemv(input)

            if self.k_chunk == 0:
                return original_output

            # generate non-repeating random indices
            random_indices = torch.randperm(self.in_features, device='cuda', dtype=torch.int32)
            
            # copy random indices to the buffer
            self.int32_buffer[:self.k] = random_indices[:self.k]

            decdec_ext.unfused_dec(
                self.dec_config,
                input,
                original_output,
            )

            return original_output


    def forward_random_ppl(self, input):
        batch_size, seq_len, hidden_size = input.size()
        assert batch_size == 1, "Batch size must be 1"
        # Decode phase
        original_output = self.gemm(input)

        if self.k_chunk == 0:
            return original_output

        for seq_idx in range(seq_len):
            # generate non-repeating random indices
            random_indices = torch.randperm(self.in_features, device='cuda', dtype=torch.int32)
            # copy random indices to the buffer
            self.int32_buffer[:self.k] = random_indices[:self.k]
            decdec_ext.unfused_dec(
                self.dec_config,
                input[0][seq_idx].unsqueeze(0).unsqueeze(0),
                original_output[0][seq_idx].unsqueeze(0).unsqueeze(0),
            )

        return original_output

    def forward(self, input):
        if self.forward_mode == 'decdec':
            return self.forward_decdec(input)
        elif self.forward_mode == 'topk':
            return self.forward_topk(input)
        elif self.forward_mode == 'ppl':
            return self.forward_ppl(input)
        elif self.forward_mode == 'topk_ppl':
            return self.forward_topk_ppl(input)
        elif self.forward_mode == 'static':
            return self.forward_static(input)
        elif self.forward_mode == 'static_ppl':
            return self.forward_static_ppl(input)
        elif self.forward_mode == 'random':
            return self.forward_random(input)
        elif self.forward_mode == 'random_ppl':
            return self.forward_random_ppl(input)
        elif self.forward_mode == 'topk_check_ppl':
            return self.forward_topk_check_ppl(input)
        elif self.forward_mode == 'exact_decdec':
            return self.forward_exact_decdec(input)
        elif self.forward_mode == 'exact_decdec_ppl':
            return self.forward_exact_decdec_ppl(input)
        else:
            raise ValueError(f"Invalid forward mode {self.forward_mode}")


def lutgemm_dequant(binary, alpha, q_bias, bitwidth, group_size):
    """
    Dequantizes the output from a LUT GEMM with reduced memory usage using FP16.
    """
    device = binary.device

    # Get shapes
    G_bin, B, O = binary.shape  # binary: [input_size // 32, bitwidth, output_size]
    input_size = G_bin * 32
    num_groups = input_size // group_size

    # Unpack bits from binary tensor
    shifts = torch.arange(32, dtype=torch.int32, device=device)
    binary_expanded = binary.unsqueeze(-1)  # [G_bin, B, O, 1]
    bits = ((binary_expanded >> shifts) & 1).byte()
    bits = bits.permute(0, 3, 1, 2).reshape(input_size, B, O)

    # In-place conversion of bits to delta, now using half precision
    delta = bits.to(torch.half).mul_(2).sub_(1)  # [input_size, bitwidth, output_size], FP16

    # Compute group indices without extra tensors
    group_indices = torch.div(
        torch.arange(input_size, device=device), group_size, rounding_mode='floor'
    ).long()  # [input_size]

    # Gather alpha and q_bias values per input dimension (already in FP16)
    alpha_values = alpha[group_indices, :, :]  # [input_size, bitwidth, output_size], FP16
    q_bias_values = q_bias[group_indices, :]  # [input_size, output_size], FP16

    # In-place multiplication and summing, keeping everything in FP16
    delta.mul_(alpha_values)
    delta_alpha_sum = delta.sum(dim=1)  # [input_size, output_size], FP16

    # In-place addition of q_bias and transpose to avoid extra memory allocation
    return (delta_alpha_sum + q_bias_values).t()  # [output_size, input_size], FP16


def sqllm_dequant(qweights, lut, bitwidth):
    """
    Restores the original FP16 weights from packed quantized weights and lookup table.
    """
    qweights = qweights.t().int()  # Shape becomes (N, num_int32s)
    N, num_int32s = qweights.shape
    device = qweights.device

    if bitwidth == 4:
        K = num_int32s * 8  # 8 weights per int32
        shifts = torch.arange(8, dtype=torch.int32, device=device) * 4  # [0, 4, ..., 28]

        # Expand and extract indices in a single step
        indices = ((qweights.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF).reshape(N, K)

        # Retrieve weights from LUT
        weights = lut.gather(1, indices.to(torch.int64))
        return weights.half()

    elif bitwidth == 3:
        group_count = num_int32s // 3  # Number of groups
        K = group_count * 32
        qweights_grouped = qweights.reshape(N, group_count, 3)  # (N, group_count, 3)

        first_int32 = qweights_grouped[:, :, 0]  # (N, group_count)
        second_int32 = qweights_grouped[:, :, 1]
        third_int32 = qweights_grouped[:, :, 2]

        # Precompute shifts tensors once
        shifts_w0_w9 = torch.arange(10, dtype=torch.int32, device=device) * 3
        shifts_w11_w20 = shifts_w0_w9 + 1  # [1, 4, ..., 28]
        shifts_w22_w31 = shifts_w0_w9 + 2  # [2, 5, ..., 29]

        # Process w0 to w9
        indices_w0_w9 = ((first_int32.unsqueeze(-1) >> shifts_w0_w9.view(1, 1, 10)) & 0x7)

        # Process w10
        idx_w10 = (((first_int32 >> 30) & 0x3) | ((second_int32 & 0x1) << 2)).unsqueeze(-1)

        # Process w11 to w20
        indices_w11_w20 = ((second_int32.unsqueeze(-1) >> shifts_w11_w20.view(1, 1, 10)) & 0x7)

        # Process w21
        idx_w21 = (((second_int32 >> 31) & 0x1) | ((third_int32 & 0x3) << 1)).unsqueeze(-1)

        # Process w22 to w31
        indices_w22_w31 = ((third_int32.unsqueeze(-1) >> shifts_w22_w31.view(1, 1, 10)) & 0x7)

        # Concatenate indices, using a single cat operation
        indices_group = torch.cat([
            indices_w0_w9,
            idx_w10,
            indices_w11_w20,
            idx_w21,
            indices_w22_w31
        ], dim=-1)  # (N, group_count, 32)

        indices = indices_group.reshape(N, K)  # (N, K)
        weights = lut.gather(1, indices.to(torch.int64))
        return weights.half()

    else:
        raise ValueError("Unsupported bitwidth. Only 3 and 4 are supported.")

def anyprec_dequant(qweights, lut, bitwidth):
    return decdec_ext.anyprec_dequant(qweights, lut, bitwidth)


def replace_linear_layers(model):
    layers_name, layers = get_layers(model)
    module_names = get_module_names(model)

    for l_idx, layer in enumerate(tqdm(layers, desc="Replacing linear layers")):
        for module_name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                custom_linear = CustomLinear(
                    module.in_features,
                    module.out_features,
                )
                # Replace the module
                parent_module = layer
                name_parts = module_name.split('.')
                for name_part in name_parts[:-1]:
                    parent_module = getattr(parent_module, name_part)
                setattr(parent_module, name_parts[-1], custom_linear)


def set_model_exact_residuals(model, exact_residuals, mask=None):
    layers_name, layers = get_layers(model)
    for l_idx, layer in enumerate(tqdm(layers, desc="Setting exact residuals")):
        for module_name, module in layer.named_modules():
            if mask is not None and (l_idx, module_name) not in mask:
                continue
            if isinstance(module, CustomLinear):
                module.exact_residuals = exact_residuals[l_idx][module_name].cuda()


def set_model_algo(model, algo):
    layers_name, layers = get_layers(model)
    for layer in layers:
        for module in layer.modules():
            if isinstance(module, CustomLinear):
                module.algo = algo


def set_model_bitwidth(model, bitwidth, mask=None):
    layers_name, layers = get_layers(model)
    for l_idx, layer in enumerate(tqdm(layers, desc="Setting bitwidth")):
        for module_name, module in layer.named_modules():
            if mask is not None and (l_idx, module_name) not in mask:
                continue
            if isinstance(module, CustomLinear):
                if isinstance(bitwidth, int):
                    module.bitwidth = bitwidth
                else:  # Mixed Precision!
                    assert module.algo == 'lutgemm', "Only LUTGEMM mixed precision directly loads 3.5-bit checkpoint"
                    module.bitwidth = module.binary.shape[1]  # retrieve the bitwidth from tensor shape


# Context manager to silence stdout
@contextlib.contextmanager
def silence_stdout():
    # Save the original stdout file descriptor (1)
    original_stdout_fd = sys.stdout.fileno()

    # Duplicate it to save the original for restoring later
    saved_stdout_fd = os.dup(original_stdout_fd)

    try:
        # Redirect stdout to /dev/null
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, original_stdout_fd)
        os.close(devnull_fd)
        
        # Yield control to the block
        yield
    finally:
        # Restore original stdout file descriptor
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.close(saved_stdout_fd)


def set_model_scale_and_cheatsheet(model, scale_and_cheatsheet, mask=None):
    layers_name, layers = get_layers(model)
    module_names = get_module_names(model)
    first_replacement = True
    for l_idx, layer in enumerate(tqdm(layers, desc="Setting cheatsheet")):
        for module_name, module in layer.named_modules():
            if mask is not None and (l_idx, module_name) not in mask:
                continue
            if isinstance(module, CustomLinear):
                cheatsheet = scale_and_cheatsheet[l_idx][module_name]['cheatsheet']
                reordered_scales = scale_and_cheatsheet[l_idx][module_name]['reordered_scales']
                module.cheatsheet = cheatsheet.cuda()
                module.reordered_scales = reordered_scales.to(DTYPE).cuda()

                # Create decdec context and config
                # Note that each CustomLinear module has its own buffers
                module.dec_context = decdec_ext.create_dec_context(32, module.int32_buffer, module.half_buffer)
                # One warning is enough, silence the rest
                with silence_stdout() if not first_replacement else contextlib.nullcontext():
                    module.dec_config = decdec_ext.create_dec_config(module.dec_context, 0, module.cheatsheet, module.reordered_scales, module.thresholds)
                first_replacement = False


def set_model_thresholds(model, thresholds, mask=None):
    layers_name, layers = get_layers(model)
    module_names = get_module_names(model)

    for l_idx, layer in enumerate(tqdm(layers, desc="Setting thresholds")):
        for module_name, module in layer.named_modules():
            if mask is not None and (l_idx, module_name) not in mask:
                continue
            if isinstance(module, CustomLinear):
                threshold_tensor = thresholds[l_idx][module_name]
                module.thresholds = threshold_tensor.to(DTYPE)


def set_model_forward_mode(model, forward_mode):
    layers_name, layers = get_layers(model)
    for layer in layers:
        for module in layer.modules():
            if isinstance(module, CustomLinear):
                module.forward_mode = forward_mode


def set_model_static_topk(model, static, mask=None):
    layers_name, layers = get_layers(model)
    for l_idx, layer in enumerate(tqdm(layers, desc="Setting static topk")):
        for module_name, module in layer.named_modules():
            if mask is not None and (l_idx, module_name) not in mask:
                continue
            if isinstance(module, CustomLinear):
                module.static_topk = static[l_idx][module_name]['sorted_indices'].to(torch.int32).cuda()


def get_average_intersection_ratio(model):
    # get all the intersection ratios, averaged over all layers
    total = 0
    count = 0
    layers_name, layers = get_layers(model)
    for layer in layers:
        for module in layer.modules():
            if isinstance(module, CustomLinear):
                total += sum(module.topk_intersection_ratio)
                count += len(module.topk_intersection_ratio)

    return total / count

def set_model_k_chunk_by_module(model, xyzws, bitwidth):
    if isinstance(bitwidth, int):
        assert len(xyzws) == 1, "Only one set of xyzw values is supported"
        xyzw_by_bitwidth = {
            bitwidth: xyzws[0]
        }
    else:
        assert bitwidth % 1 == 0.5, f"Only x.5-bit mixed precision is supported, got {bitwidth}"
        assert len(xyzws) == 2, "Only two sets of xyzw values are supported"
        low_bitwidth = int(bitwidth)
        high_bitwidth = low_bitwidth + 1
        xyzw_by_bitwidth = {
            low_bitwidth: xyzws[0],
            high_bitwidth: xyzws[1]
        }

    # Determine model type based on class name
    model_class_name = model.__class__.__name__.lower()
    if 'llama' in model_class_name:
        model_type = 'llama'
        k_chunk_template = {
            'self_attn.q_proj': 0,
            'self_attn.k_proj': 0,
            'self_attn.v_proj': 0,
            'self_attn.o_proj': 1,
            'mlp.gate_proj': 2,
            'mlp.up_proj': 2,
            'mlp.down_proj': 3
        }
    elif 'phi' in model_class_name:
        model_type = 'phi'
        k_chunk_template = {
            'self_attn.qkv_proj': 0,
            'self_attn.o_proj': 1,
            'mlp.gate_up_proj': 2,
            'mlp.down_proj': 3
        }
    else:
        raise ValueError(f"Model not recognized: {model_class_name}")

    layers_name, layers = get_layers(model)
    for layer in layers:
        for module_name, module in layer.named_modules():
            if isinstance(module, CustomLinear) and module.bitwidth in xyzw_by_bitwidth:
                xyzw = xyzw_by_bitwidth[module.bitwidth]
                k_chunk_index = k_chunk_template.get(module_name)
                if k_chunk_index is not None:
                    module.k_chunk = xyzw[k_chunk_index]
                    # update dec_config
                    decdec_ext.update_dec_config(module.dec_config, module.dec_context, module.k_chunk)
    print(f"Set model {model_type} to {(xyzw_by_bitwidth)}")



def load_fake_model(model_path):
    print("Loading model from path:", model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path).to(DTYPE)

    print("Loading thresholds and cheatsheet...")
    thresholds = torch.load(model_path + '/thresholds.pt')
    scale_and_cheatsheet = torch.load(model_path + '/cheatsheet.pt')

    replace_linear_layers(model)
    set_model_thresholds(model, thresholds)
    set_model_scale_and_cheatsheet(model, scale_and_cheatsheet)
    set_model_algo(model, 'fake')

    print("Moving model to GPU...")

    model = model.eval().cuda()

    return model


def get_module_from_l_idx_and_name(model, l_idx, module_name):
    layers_name, layers = get_layers(model)
    layer = layers[l_idx]
    for name, module in layer.named_modules():
        if name == module_name:
            return module
    raise ValueError(f"Module {module_name} not found")


def overwrite_model_state_dict(model, state_dict, mask=None):
    """Forcefully assign every parameter in the state_dict to the model."""
    for name, param in state_dict.items():
        name_parts = name.split('.')
        module = model
        for part in name_parts[:-1]:
            module = getattr(module, part)

        if mask is not None and name.startswith('model.layers.'):
            l_idx = int(name_parts[2])
            module_name = name_parts[3] + '.' + name_parts[4]
            if name_parts[3] in ('self_attn', 'mlp') and (l_idx, module_name) not in mask:
                # If this is on a layer that is not in the mask, skip
                # Note that modules like model.norm.weight, lm_head.weight, 
                # model.layers.5.post_attention_layernorm.weight etc. should not be skipped
                continue

        # if dtype of tensor is bf16 or float, convert to DTYPE
        if param.dtype in [torch.bfloat16, torch.float32]:
            param = param.to(DTYPE)
        param = param.cuda()
        # delete the original parameter
        if hasattr(module, name_parts[-1]):
            delattr(module, name_parts[-1])
        setattr(module, name_parts[-1], param)


def load_empty_model(base_model_repo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(base_model_repo, torch_dtype=DTYPE)
    replace_linear_layers(model)
    return model


def load_model_common(model, model_path, bitwidth, load_exact_residuals, mask=None, residual_bitwidth=None):
    state_dict = torch.load(f'{model_path}/pytorch_model.bin')
    # overwrite the model with the state_dict
    overwrite_model_state_dict(model, state_dict, mask=mask)

    # load thresholds and cheatsheet
    thresholds = torch.load(f'{model_path}/thresholds.pt')
    scale_and_cheatsheet = torch.load(f'{model_path}/cheatsheet.pt')
    try:
        static = torch.load(f'{model_path}/static.pt')
    except FileNotFoundError:
        print("No static file found, skipping setup.")
        static = None

    if load_exact_residuals:
        if residual_bitwidth in (2, 8):
            residuals_path = f'{model_path}/quantized_residuals_{residual_bitwidth}bit.pt'
            quantized_residuals = torch.load(residuals_path)
            set_model_exact_residuals(model, quantized_residuals, mask=mask)
        else:
            assert residual_bitwidth == 16 or residual_bitwidth is None, "Unsupported residual bitwidth"
            exact_residuals = torch.load(f'{model_path}/exact_residuals.pt')
            set_model_exact_residuals(model, exact_residuals, mask=mask)

    set_model_thresholds(model, thresholds, mask=mask)
    set_model_scale_and_cheatsheet(model, scale_and_cheatsheet, mask=mask)
    set_model_bitwidth(model, bitwidth, mask=mask)

    if static is not None:
        set_model_static_topk(model, static, mask=mask)


def bitwidth_map_to_mask(bitwidth_map, bitwidth):
    mask = set(
        (l_idx, module_name) 
        for l_idx, module_name in bitwidth_map 
        if bitwidth_map[(l_idx, module_name)] == bitwidth
    )

    # if 'self_attn.qkv_proj' is in the mask, add 'self_attn.q_proj', 'self_attn.k_proj', and 'self_attn.v_proj' to the mask
    # if 'mlp.gate_up_proj' is in the mask, add 'mlp.gate_proj' and 'mlp.up_proj' to the mask

    additional_mask = set()
    for l_idx, module_name in mask:
        match module_name:
            case 'self_attn.qkv_proj':
                additional_mask.add((l_idx, 'self_attn.q_proj'))
                additional_mask.add((l_idx, 'self_attn.k_proj'))
                additional_mask.add((l_idx, 'self_attn.v_proj'))
            case 'mlp.gate_up_proj':
                additional_mask.add((l_idx, 'mlp.gate_proj'))
                additional_mask.add((l_idx, 'mlp.up_proj'))
    mask.update(additional_mask)

    return mask

def load_kl_bitwidth_map(base_model_repo, algo, bitwidth):
    assert bitwidth == 3.5, "Only 3.5-bit supported for now"
    assert algo in ('anyprec', 'sqllm'), "Only anyprec and sqllm supported for this mode of mixed precision"
    if algo == 'anyprec':
        algo = 'sqllm'
    kl_path_3 = f'./kl_div/{algo}-({base_model_repo.split("/")[-1]})-w3.json'
    kl_path_4 = f'./kl_div/{algo}-({base_model_repo.split("/")[-1]})-w4.json'
    with open(kl_path_3, 'r') as f:
        kl_div_3 = json.load(f)
    with open(kl_path_4, 'r') as f:
        kl_div_4 = json.load(f)
    
    # Take the difference of KL divs
    kl_diff = {int(k): kl_div_3[k] - kl_div_4[k] for k in kl_div_3}
    # Sort the keys by the difference
    sorted_layer_indices = sorted(kl_diff, key=kl_diff.get)
    # The smaller half of the keys will be 3-bit, the larger half will be 4-bit
    bitwidth_layer_map = {k: 3 if i < len(sorted_layer_indices) // 2 else 4 for i, k in enumerate(sorted_layer_indices)}
    # check that half of the keys are 3-bit and half are 4-bit
    assert len([k for k in bitwidth_layer_map if bitwidth_layer_map[k] == 3]) == len(sorted_layer_indices) // 2
    assert len([k for k in bitwidth_layer_map if bitwidth_layer_map[k] == 4]) == len(sorted_layer_indices) // 2

    # Now convert the layer-wise map to a module-wise map

    bitwidth_map = {}

    module_names = [
        'self_attn.qkv_proj',
        'self_attn.o_proj',
        'mlp.gate_up_proj',
        'mlp.down_proj',
    ]

    for l_idx, bitwidth in bitwidth_layer_map.items():
        for module_name in module_names:
            bitwidth_map[(l_idx, module_name)] = bitwidth

    return bitwidth_map


def load_model(base_model_repo, model_paths, algo, bitwidth, load_exact_residuals=False, bitwidth_map=None, residual_bitwidth=None):
    match bitwidth:
        case 3:
            bitwidths_to_load = [3]
            bitwidth_map = None  # ignore bitwidth_map for non-mixed precision models
        case 4:
            bitwidths_to_load = [4]
            bitwidth_map = None  # ignore bitwidth_map for non-mixed precision models
        case 3.5:
            match algo:
                case 'lutgemm':
                    bitwidths_to_load = [3.5]
                case 'sqllm' | 'anyprec':
                    bitwidths_to_load = [3, 4]
                    print("Loading mixed precision model")
                    assert bitwidth_map is not None, "bitwidth_map must be provided for mixed precision model"
                    if bitwidth_map == 'kl':
                        print("Loading KL divergence based bitwidth map...")
                        bitwidth_map = load_kl_bitwidth_map(base_model_repo, algo, bitwidth)
        case _:
            raise ValueError(f"Invalid bitwidth: {bitwidth}")

    model = load_empty_model(base_model_repo)
    set_model_algo(model, algo)
    
    for bw in bitwidths_to_load:
        print(f"==> Loading {bw}-bit weights <==")
        model_path = model_paths[bw]
        if algo in ('sqllm', 'anyprec') and bitwidth_map is not None:
            mask = bitwidth_map_to_mask(bitwidth_map, bw)
        else:
            mask = None
        load_model_common(model, model_path, bw, load_exact_residuals, mask=mask, residual_bitwidth=residual_bitwidth)

    return model.eval().cuda()


def run_eval_common(base_model_repo, model_paths, algorithms, configs, forward_modes, tasks=None, datasets=None, eval_type='lm_eval', bitwidth_map=None, residual_bitwidth=None):
    model_name = base_model_repo.split('/')[-1]
    previously_loaded_model = None
    previously_loaded_model_tuple = None
    i = 0
    for config, algo, forward_mode in product(configs, algorithms, forward_modes):
        print("===============================================")
        print("Model:", model_name)
        print("Algo:", algo)
        print("Forward mode:", forward_mode)
        if residual_bitwidth is not None:
            print("Residual bitwidth:", residual_bitwidth)
        print("Config:", config)
        print("Progress:", f"{i+1}/{len(configs) * len(algorithms) * len(forward_modes)}")
        print("===============================================")
        i += 1
        name, bitwidth, *xyzws = config

        if isinstance(bitwidth, int):
            assert len(xyzws) == 1, "Provide single xyzw config for fixed bitwidth"
            xyzw = xyzws[0]
            xyzw_string = ' '.join(map(str, xyzw))
        else:
            assert len(xyzws) == 2, "Provide two xyzw configs for mixed precision"
            xyzw_low, xyzw_high = xyzws
            xyzw_string = ' '.join(map(str, xyzw_low)) + '_' + ' '.join(map(str, xyzw_high))

        results_dir = './results_' + eval_type
        os.makedirs(results_dir, exist_ok=True)

        if residual_bitwidth is None:
            residual_bitwidth_str = ''
        else:
            residual_bitwidth_str = f'_r{residual_bitwidth}'

        out_path = f"{results_dir}/{forward_mode + residual_bitwidth_str}/{model_name}_{algo}_{bitwidth}_{forward_mode + residual_bitwidth_str}_{name}_{xyzw_string}.out"

        if os.path.exists(out_path):
            print("Results already exist. Skipping...")
            continue

        # ==================== LOAD MODEL ====================
        if previously_loaded_model_tuple != (base_model_repo, algo, bitwidth):
            # clear memory
            if previously_loaded_model is not None:
                del previously_loaded_model
                del model
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            print("Loading model...")
            load_exact_residuals = 'exact' in forward_mode
            model = load_model(base_model_repo, model_paths, algo, bitwidth, load_exact_residuals=load_exact_residuals, bitwidth_map=bitwidth_map, residual_bitwidth=residual_bitwidth)
            tokenizer = AutoTokenizer.from_pretrained(base_model_repo)
            previously_loaded_model = model
            previously_loaded_model_tuple = (base_model_repo, algo, bitwidth)

        else:
            model = previously_loaded_model
            tokenizer = AutoTokenizer.from_pretrained(base_model_repo)
        
        # ==================== Configure model ====================
        set_model_k_chunk_by_module(model, xyzws, bitwidth)
        set_model_forward_mode(model, 'decdec')

        # ==================== Inference test ====================
        # Sample input text
        input_text = "Question: Explain how to derive Newton's shell theorem.\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        # Create a streamer for real-time text generation
        streamer = TextStreamer(tokenizer)

        # Perform inference with streaming
        with torch.no_grad():
            model.generate(input_ids, max_length=300, num_return_sequences=1, streamer=streamer)

        # clean up memory
        del streamer
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # ==================== EVALUATE ====================
        set_model_forward_mode(model, forward_mode)

        match eval_type:
            case 'lm_eval':
                print("-------------------- LM-EVAL --------------------")
                results = run_lm_eval(tokenizer, model, tasks)
            case 'perplexity':
                print("-------------------- PPL-EVAL --------------------")
                results = evaluate_ppl(model, tokenizer, datasets, verbose=True, chunk_size=2048, tokenizer_type=None)
                if forward_mode == 'topk_check_ppl':
                    avg_intersection_ratio = get_average_intersection_ratio(model)
                    results['avg_intersection_ratio'] = avg_intersection_ratio
            case _:
                raise ValueError(f"Invalid eval type: {eval_type}")

        # Add bitwidth_map as string to results if using mixed precision
        if not isinstance(bitwidth, int):
            results['bitwidth_map'] = str(bitwidth_map)

        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, 'w') as f:
            f.write(json.dumps(results, indent=4))


def run_eval(base_model_repo, model_paths, algorithms, tasks, configs, forward_modes, bitwidth_map=None, residual_bitwidth=None):
    run_eval_common(base_model_repo, model_paths, algorithms, configs, forward_modes, tasks=tasks, eval_type='lm_eval', bitwidth_map=bitwidth_map, residual_bitwidth=residual_bitwidth)


def run_eval_ppl(base_model_repo, model_paths, algorithms, datasets, configs, forward_modes, bitwidth_map=None, residual_bitwidth=None):
    run_eval_common(base_model_repo, model_paths, algorithms, configs, forward_modes, datasets=datasets, eval_type='perplexity', bitwidth_map=bitwidth_map, residual_bitwidth=residual_bitwidth)


def run_eval_fp16(base_model_repo, tasks):
    results_dir = './results_lm_eval'
    os.makedirs(results_dir, exist_ok=True)
    out_path = f"{results_dir}/{base_model_repo.split('/')[-1]}_fp16.out"

    if os.path.exists(out_path):
        print("Results already exist. Skipping...")
        return

    model = AutoModelForCausalLM.from_pretrained(base_model_repo).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(base_model_repo)

    results = run_lm_eval(tokenizer, model, tasks)

    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=4))


def run_eval_ppl_fp16(base_model_repo, datasets):
    results_dir = './results_perplexity'
    os.makedirs(results_dir, exist_ok=True)
    out_path = f"{results_dir}/{base_model_repo.split('/')[-1]}_fp16.out"

    if os.path.exists(out_path):
        print("Results already exist. Skipping...")
        return

    model = AutoModelForCausalLM.from_pretrained(base_model_repo).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(base_model_repo)

    results = evaluate_ppl(model, tokenizer, datasets, verbose=True, chunk_size=2048, tokenizer_type=None)

    with open(out_path, 'w') as f:
        f.write(json.dumps(results, indent=4))


def append_mixed_precision_configs(configs):
    configs_3 = [config for config in configs if config[1] == 3]
    configs_4 = [config for config in configs if config[1] == 4]
    assert len(configs_3) == len(configs_4), "Number of 3-bit and 4-bit configs must be equal"

    # Each config looks like (name, bitwidth, xyzw)
    # For 3.5-bit mixed precision, we need to pair up the 3-bit and 4-bit configs
    # Match by name

    for config_3 in configs_3:
        name_3, bitwidth_3, xyzw_3 = config_3
        for config_4 in configs_4:
            name_4, bitwidth_4, xyzw_4 = config_4
            if name_3 == name_4:
                configs.append((name_3, 3.5, xyzw_3, xyzw_4))
                break
