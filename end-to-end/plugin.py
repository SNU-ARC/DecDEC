import torch
import decdec_ext 

"""
Any-Precision
"""
@torch.library.custom_op("plugin::anyprec_gemv", mutates_args={"output"})
def anyprec_gemv(x: torch.Tensor, output:torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, bitwidth:int) -> None:
    decdec_ext.anyprec_gemv(x, output, q_weight, lut, bitwidth)

@anyprec_gemv.register_fake
def _(x, output, q_weight, lut, bitwidth):
    return None

@torch.library.custom_op("plugin::dec_anyprec", mutates_args={"output"})
def dec_anyprec(dec_config: int, x: torch.Tensor, output:torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, bitwidth:int) -> None:
    decdec_ext.dec_anyprec(dec_config, x, output, q_weight, lut, bitwidth)

@dec_anyprec.register_fake
def _(dec_config, x, output, q_weight, lut, bitwidth):
    return None

"""
LUTGEMM
"""
@torch.library.custom_op("plugin::lutgemm_gemv", mutates_args={"output"})
def lutgemm_gemv(x: torch.Tensor, output: torch.Tensor, q_weight: torch.Tensor, alpha: torch.Tensor, q_bias: torch.Tensor, bitwidth: int, group_size: int) -> None:
    decdec_ext.lutgemm_gemv(x, output, q_weight, alpha, q_bias, bitwidth, group_size)

@lutgemm_gemv.register_fake
def _(x, output, q_weight, alpha, q_bias, bitwidth, group_size):
    return None

@torch.library.custom_op("plugin::dec_lutgemm", mutates_args={"output"})
def dec_lutgemm(dec_config: int, x: torch.Tensor, output: torch.Tensor, q_weight: torch.Tensor, alpha: torch.Tensor, q_bias: torch.Tensor, bitwidth: int, group_size: int) -> None:
    decdec_ext.dec_lutgemm(dec_config, x, output, q_weight, alpha, q_bias, bitwidth, group_size)

@dec_lutgemm.register_fake
def _(dec_config, x, output, q_weight, alpha, q_bias, bitwidth, group_size):
    return None


"""
DECConfig, DECContext
"""

@torch.library.custom_op("plugin::create_dec_context", mutates_args={})
def create_dec_context(n_tb: int, index_buffer: torch.Tensor, activation_buffer: torch.Tensor) -> int:
    dec_context = decdec_ext.create_dec_context(n_tb, index_buffer, activation_buffer)
    return dec_context

@create_dec_context.register_fake
def _(n_tb, index_buffer, activation_buffer):
    return None

@torch.library.custom_op("plugin::create_dec_config", mutates_args={})
def create_dec_config(dec_context: int, k_chunk: int, q_residual: torch.Tensor, scales: torch.Tensor, thresholds: torch.Tensor) -> int:
    dec_config = decdec_ext.create_dec_config(dec_context, k_chunk, q_residual, scales, thresholds)
    return dec_config

@create_dec_config.register_fake
def _(dec_context, k_chunk, q_residual, scales, thresholds):
    return None

@torch.library.custom_op("plugin::update_dec_config", mutates_args={})
def update_dec_config(dec_config: int, dec_context: int, k_chunk: int) -> None:
    decdec_ext.update_dec_config(dec_config, dec_context, k_chunk)

@update_dec_config.register_fake
def _(dec_config, dec_context, k_chunk):
    return None
