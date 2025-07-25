# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import transformers
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from plugin import *

from tqdm import tqdm

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None
    model_name: Optional[str] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        assert name in transformer_configs, f"Unknown model name: {name}, available: {transformer_configs.keys()}"
        return cls(**transformer_configs[name])

transformer_configs = {
    "Meta-Llama-3-8B-Instruct": dict(model_name="Meta-Llama-3-8B-Instruct", block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "Meta-Llama-3-70B-Instruct": dict(model_name="Meta-Llama-3-70B-Instruct", block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "Phi-3-medium-4k-instruct": dict(model_name='Phi-3-medium-4k-instruct', block_size=4096, n_layer=40, n_head=40, n_local_heads=10, dim=5120, intermediate_size=17920, vocab_size=32064, rope_base=10000),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.half):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        #cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, dtype, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None, halve_layers=False, bitwidth_map=None) -> None:

        linear_kwargs_per_layer = [linear_kwargs.copy() for _ in range(config.n_layer)]

        # if bitwidth_map is not None, set the bitwidth for each layer for mixed-precision inference
        if bitwidth_map is not None:
            for l_idx, bitwidth in bitwidth_map.items():
                linear_kwargs_per_layer[l_idx]["bitwidth"] = bitwidth

        super().__init__()
        self.config = config
        self.dtype = dtype

        # if halve_layers, halve the number of layers for testing purposes
        if halve_layers:
            config.n_layer = config.n_layer // 2

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, linear_class, linear_kwargs_per_layer[l_idx]) for l_idx in range(config.n_layer))
        if "phi" in config.model_name.lower():
            self.norm = Phi3RMSNorm(config.dim, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.cache_initialized = False

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.cache_initialized = True


    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.cache_initialized, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    def load_dec_data(self, dec_data):
        for idx, layer in enumerate(self.layers):
            prefix = f"layers.{idx}."
            layer.attention.wqkv.load_dec_data(
                                dec_data[prefix+"attention.wqkv.q_residual"],
                                dec_data[prefix+"attention.wqkv.scales"],
                                dec_data[prefix+"attention.wqkv.thresholds"]
                                )
            layer.attention.wo.load_dec_data(
                                dec_data[prefix+"attention.wo.q_residual"],
                                dec_data[prefix+"attention.wo.scales"],
                                dec_data[prefix+"attention.wo.thresholds"]
                                )
            layer.feed_forward.w1w3.load_dec_data(
                                dec_data[prefix+"feed_forward.w1w3.q_residual"],
                                dec_data[prefix+"feed_forward.w1w3.scales"],
                                dec_data[prefix+"feed_forward.w1w3.thresholds"]
                                )
            layer.feed_forward.w2.load_dec_data(
                                dec_data[prefix+"feed_forward.w2.q_residual"],
                                dec_data[prefix+"feed_forward.w2.scales"],
                                dec_data[prefix+"feed_forward.w2.thresholds"]
                                )

    def create_dec_context(self, n_tbs, buffer_size = 1024):
        if isinstance(n_tbs, int):
            n_tbs = [n_tbs]
        self.selected_rows_buffers = [torch.empty(buffer_size, dtype=torch.int, device='cuda') for _ in range(len(n_tbs))]
        self.selected_activations_buffers = [torch.empty(buffer_size, dtype=self.dtype, device='cuda') for _ in range(len(n_tbs))]
        self.dec_contexts = [create_dec_context(n_tb, self.selected_rows_buffers[i], self.selected_activations_buffers[i]) for i, n_tb in enumerate(n_tbs)]

    def update_dec_context(self, n_tbs):
        if isinstance(n_tbs, int):
            n_tbs = [n_tbs]
        assert len(self.dec_contexts) == len(n_tbs)
        for i, n_tb in enumerate(n_tbs):
            self.dec_contexts[i] = create_dec_context(n_tb, self.selected_rows_buffers[i], self.selected_activations_buffers[i])

    def set_dec_config(self, k_chunks):
        # k_chunks: [qkv, o, gate/up, down], repeated twice for mixed-precision
        k_chunks_per_bitwidth = {}
        if len(k_chunks) == 8:
            k_chunks_per_bitwidth[3] = [k_chunks[0], k_chunks[1], k_chunks[2], k_chunks[3]]
            k_chunks_per_bitwidth[4] = [k_chunks[4], k_chunks[5], k_chunks[6], k_chunks[7]]
        else:
            assert len(k_chunks) == 4
            k_chunks_per_bitwidth[3] = k_chunks
            k_chunks_per_bitwidth[4] = k_chunks

        # set the dec_config for each layer, using the values relevant to the bitwidth
        for layer in self.layers:
            for i, linear in enumerate([layer.attention.wqkv, layer.attention.wo, layer.feed_forward.w1w3, layer.feed_forward.w2]):
                bitwidth = linear.bitwidth
                # fix this mess, this was a quick patch for 3.5-bit inference
                linear.create_dec_config(self.dec_contexts[0 if (len(self.dec_contexts) == 1 or bitwidth == 3) else 1], k_chunks_per_bitwidth[bitwidth][i])

    def update_dec_config(self, k_chunks):
        # k_chunks: [qkv, o, gate/up, down], repeated twice for mixed-precision
        k_chunks_per_bitwidth = {}
        if len(k_chunks) == 8:
            k_chunks_per_bitwidth[3] = [k_chunks[0], k_chunks[1], k_chunks[2], k_chunks[3]]
            k_chunks_per_bitwidth[4] = [k_chunks[4], k_chunks[5], k_chunks[6], k_chunks[7]]
        else:
            assert len(k_chunks) == 4
            k_chunks_per_bitwidth[3] = k_chunks
            k_chunks_per_bitwidth[4] = k_chunks

        # update the dec_config for each layer, using the values relevant to the bitwidth
        for layer in self.layers:
            for i, linear in enumerate([layer.attention.wqkv, layer.attention.wo, layer.feed_forward.w1w3, layer.feed_forward.w2]):
                bitwidth = linear.bitwidth
                linear.update_dec_config(self.dec_contexts[0 if (len(self.dec_contexts) == 1 or bitwidth == 3) else 1], k_chunks_per_bitwidth[bitwidth][i])

    @classmethod
    def from_name(cls, dtype, name: str, linear_class=nn.Linear, linear_kwargs=None, halve_layers=False, bitwidth_map=None) -> "Transformer":
        return cls(dtype, ModelArgs.from_name(name), linear_class=linear_class, linear_kwargs=linear_kwargs, halve_layers=halve_layers, bitwidth_map=bitwidth_map)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None) -> None:
        super().__init__()
        self.attention = Attention(config, linear_class, linear_kwargs)
        self.feed_forward = FeedForward(config, linear_class, linear_kwargs)

        if "llama" in config.model_name.lower():
            self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
            self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)
            self.pre_feedforward_layernorm = None
            self.post_feedforward_layernorm = None
        elif "phi" in config.model_name.lower():
            self.input_layernorm = Phi3RMSNorm(config.dim, config.norm_eps)
            self.post_attention_layernorm = Phi3RMSNorm(config.dim, config.norm_eps)
            self.pre_feedforward_layernorm = None
            self.post_feedforward_layernorm = None
        else:
            raise NotImplementedError

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(
                                self.input_layernorm(x), 
                                mask, input_pos
                                )

        if self.pre_feedforward_layernorm != None:
            h = self.pre_feedforward_layernorm(h)
        
        out = self.feed_forward(self.post_attention_layernorm(h))

        if self.post_feedforward_layernorm != None:
            out = self.post_feedforward_layernorm(out)

        out = h + out 

        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None) -> None:
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = linear_class(config.dim, total_head_dim, bias=False, **(linear_kwargs or {}))
        self.wo = linear_class(config.dim, config.dim, bias=False, **(linear_kwargs or {}))

        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.config = config

        self.scaling = 1/ math.sqrt(config.head_dim)

        if "llama" in config.model_name.lower():
            self.rotary_emb = LlamaRotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=config.block_size,
                    base=config.rope_base
            )
            self.sdpa_scaling = None
        elif "phi" in config.model_name.lower():
            self.rotary_emb = Phi3RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=config.block_size,
                    base=config.rope_base
            )
            self.sdpa_scaling = None
        else:
            raise NotImplementedError

    def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        cos, sin = self.rotary_emb(v, input_pos.unsqueeze(0))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0) #, scale=self.sdpa_scaling)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        y = self.wo(y)

        return y



class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs, linear_class=nn.Linear, linear_kwargs=None) -> None:
        super().__init__()
        self.config = config
        self.w1w3 = linear_class(config.dim, config.intermediate_size*2, bias=False, **(linear_kwargs or {}))
        self.w2 = linear_class(config.intermediate_size, config.dim, bias=False, **(linear_kwargs or {}))

        self.act_fn = F.silu

    def forward(self, x: Tensor) -> Tensor:
        w1_out, w3_out = self.w1w3(x).split([self.config.intermediate_size, self.config.intermediate_size], dim=-1)
        return self.w2(self.act_fn(w1_out) * w3_out)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class LlamaRotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs)
            self.register_buffer(
                "inv_freq", inv_freq,
                persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq",
                                 self.original_inv_freq,
                                 persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float()
                     @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Phi3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

