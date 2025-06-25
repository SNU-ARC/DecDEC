#ifndef DEC_CONFIG_H
#define DEC_CONFIG_H

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "macros.h"
#include "dec_context.h"
#include "datatype.h"


#include <torch/extension.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>



class DECConfig {
public:
    DataType data_type; // Added data_type field

    DECContext* dec_context;
    uint32_t* q_residual;
    void* reordered_scales;
    void* thresholds;

    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t num_q_residual_cols;

    uint32_t num_chunks;
    uint32_t chunk_size;
    uint32_t k_chunk;
    uint32_t channels_per_tb;
    dim3 grid_dim;
    dim3 block_dim;
    uint32_t shared_memory_size;
    bool reduced_shared_memory;

    cudaEvent_t event_1;
    cudaEvent_t event_2;


    DECConfig(
        DataType _data_type,
        DECContext* _dec_context,
        uint32_t* _q_residual,
        void* _reordered_scales,
        void* _thresholds,
        uint32_t _num_rows,
        uint32_t _num_cols,
        uint32_t _num_q_residual_cols
    );

    ~DECConfig();

    void setup(uint32_t k_chunk);
};

uintptr_t create_dec_config (
    uintptr_t dec_context_uintptr,
    uint32_t k_chunk,
    torch::Tensor q_residual,
    torch::Tensor reordered_scales,
    torch::Tensor thresholds
);

std::unordered_map<std::string, int> read_dec_config(
    uintptr_t dec_config_uintptr
);

void destroy_dec_config(
    uintptr_t dec_config_uintptr
);

void update_dec_config(
    uintptr_t dec_config_uintptr,
    uintptr_t new_dec_context_uintptr,
    uint32_t k_chunk
);

#endif // DEC_CONFIG_H

