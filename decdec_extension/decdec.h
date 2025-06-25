#ifndef DECDEC_CUH
#define DECDEC_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>

#include <torch/extension.h>
#include <cuda_runtime.h>

void dec(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output
);

void unfused_dec(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output
);

void anyprec_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
);

torch::Tensor anyprec_dequant(
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
);

void lutgemm_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size 
);

void sqllm_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
);

void dec_anyprec(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
);

void dec_lutgemm(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth, 
    int group_size 
);

void dec_sqllm(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor lut,
    int bitwidth
);

void dummy_anyprec(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    int dummy_sm,
    long long dummy_iters
);

void dummy_lutgemm(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor alpha,
    torch::Tensor qbias,
    int bitwidth,
    int group_size,
    int dummy_sm,
    long long dummy_iters
);

void dummy_sqllm(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    int dummy_sm,
    long long dummy_iters
);

#endif // DECDEC_CUH
