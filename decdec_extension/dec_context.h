#ifndef DECCONTEXT_H
#define DECCONTEXT_H

#include "macros.h"
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "datatype.h"
#include "typetraits.h"

class DECContext {
public:
    DataType data_type;
    cudaStream_t sub_stream;
    cudaEvent_t event;
    uint32_t num_tb;
    uint32_t buffer_size;
    uint32_t* selected_rows_buffer;
    void* selected_activations_buffer;

    DECContext (
        DataType _data_type,
        cudaStream_t _sub_stream,
        cudaEvent_t _event,
        uint32_t _num_tb,
        uint32_t _buffer_size,
        uint32_t* _selected_rows_buffer,
        void* _selected_activations_buffer
    );

    ~DECContext();
};

uintptr_t create_dec_context (
    uint32_t num_tb,
    torch::Tensor selected_rows_buffer,
    torch::Tensor selected_activations_buffer
);

#endif
