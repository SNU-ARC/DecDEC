#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dec_context.h"
#include "typetraits.h"
#include "datatype.h"
#include <cuda_runtime.h>

DECContext::DECContext (
    DataType _data_type,
    cudaStream_t _sub_stream,
    cudaEvent_t _event,
    uint32_t _num_tb,
    uint32_t _buffer_size,
    uint32_t* _selected_rows_buffer,
    void* _selected_activations_buffer
):
    data_type(_data_type),
    sub_stream(_sub_stream),
    event(_event),
    num_tb(_num_tb),
    buffer_size(_buffer_size),
    selected_rows_buffer(_selected_rows_buffer),
    selected_activations_buffer(_selected_activations_buffer)
{}

DECContext::~DECContext() {}

template<DataType DT>
uintptr_t create_dec_context_templated (
    uint32_t num_tb,
    torch::Tensor selected_rows_buffer,
    torch::Tensor selected_activations_buffer
) {
    cudaStream_t sub_stream;
    cudaStreamCreateWithFlags(&sub_stream, cudaStreamNonBlocking);

    cudaEvent_t event;
    cudaEventCreate(&event);

    uint32_t buffer_size = selected_activations_buffer.numel();

    auto* dec_context = new DECContext(
        DT,
        sub_stream,
        event,
        num_tb,
        buffer_size,
        (uint32_t*)selected_rows_buffer.data_ptr<int>(),
        (void*) selected_activations_buffer.data_ptr<ATEN_DTYPE(DT)>()
    );
    return reinterpret_cast<uintptr_t>(dec_context);
}

uintptr_t create_dec_context (
    uint32_t num_tb,
    torch::Tensor selected_rows_buffer,
    torch::Tensor selected_activations_buffer
) {
    // Check that both are on GPU
    TORCH_CHECK(selected_rows_buffer.is_cuda(), "selected_rows_buffer must be on GPU.");
    TORCH_CHECK(selected_activations_buffer.is_cuda(), "selected_activations_buffer must be on GPU.");

    // Check that both have identical sizes
    TORCH_CHECK(selected_rows_buffer.numel() == selected_activations_buffer.numel(), "selected_rows_buffer and selected_activations_buffer must be of the same size.");

    // Check that selected_rows_buffer is of type int
    TORCH_CHECK(selected_rows_buffer.scalar_type() == at::kInt, "selected_rows_buffer must be of type int.");

    // Check that all tensors are contiguous
    TORCH_CHECK(selected_rows_buffer.is_contiguous(), "selected_rows_buffer must be contiguous.");
    TORCH_CHECK(selected_activations_buffer.is_contiguous(), "selected_activations_buffer must be contiguous.");

    auto dtype = selected_activations_buffer.scalar_type();
    if (dtype == at::kFloat) {
        return create_dec_context_templated<DataType::FP32>(
            num_tb,
            selected_rows_buffer,
            selected_activations_buffer
        );
    } else if (dtype == at::kHalf) {
        return create_dec_context_templated<DataType::FP16>(
            num_tb,
            selected_rows_buffer,
            selected_activations_buffer
        );
    } else if (dtype == at::kBFloat16) {
        return create_dec_context_templated<DataType::BF16>(
            num_tb,
            selected_rows_buffer,
            selected_activations_buffer
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}
