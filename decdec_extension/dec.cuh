#ifndef DEC_CUH
#define DEC_CUH

#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "sqllm.h"
#include "lutgemm.h"
#include "dec_config.h"
#include "dec_context.h"
#include "datatype.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>


#include <assert.h>


template<DataType DT>
__global__ void fused_dec_kernel(
    const uint32_t *__restrict__ cheatsheet,
    const FP_DTYPE(DT) *__restrict__ activations,
    uint32_t *__restrict__ selected_rows,
    FP_DTYPE(DT) *__restrict__ selected_activations,
    FP_DTYPE(DT) *__restrict__ final_sum,
    const FP_DTYPE(DT) *__restrict__ reordered_scales,
    const FP_DTYPE(DT) threshold_a,
    const FP_DTYPE(DT) threshold_b,
    const uint32_t num_rows,
    const uint32_t num_cols,
    const uint32_t num_chunks,
    const uint32_t chunk_size,
    const uint32_t k_chunk,
    const uint32_t channels_per_tb,
    const bool reduced_shared_memory
);

template<DataType DT>
__global__ void dec_kernel(
    const uint32_t *__restrict__ cheatsheet,
    const FP_DTYPE(DT) *__restrict__ activations,
    uint32_t *__restrict__ selected_rows,
    FP_DTYPE(DT) *__restrict__ final_sum,
    const FP_DTYPE(DT) *__restrict__ reordered_scales,
    const uint32_t num_rows,
    const uint32_t num_cols,
    const uint32_t num_chunks,
    const uint32_t chunk_size,
    const uint32_t k_chunk,
    const uint32_t channels_per_tb,
    const bool reduced_shared_memory
);
#endif // DEC_CUH