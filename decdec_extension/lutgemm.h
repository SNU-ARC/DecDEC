#ifndef LUTGEMM_CUH
#define LUTGEMM_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "datatype.h"
#include "typetraits.h"

#include <torch/extension.h>
#include <cuda_runtime.h>

template<DataType DT>
__global__ void nqmv_bias(
    uint32_t* W, // quantized weights, W[kSize/32][nb][mSize]
    FP_DTYPE(DT)* alpha, // alpha[num_groups][nb][mSize]
    FP_DTYPE(DT)* q_bias, // q_bias[num_groups][mSize]
    FP_DTYPE(DT)* input, // input[kSize]
    FP_DTYPE(DT)* output, // output[mSize]
    int M, // mSize
    int K, // kSize
    int NUM_BITS, // nb
    int group_size // group_size
);

#endif

