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
#include "decdec.h"
#include "dec.cuh"
#include "anyprec.h"
#include "typetraits.h"
#include "datatype.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <assert.h>

// SQLLM 
#define SQLLM_BLOCKWIDTH 128
#define SQLLM_BLOCKHEIGHT3 12
#define SQLLM_BLOCKHEIGHT4 16

// LUTGEMM
#define K_TILE_SIZE 32
#define NUM_THREADS 256
#define M_TILE_SIZE 2048

////////////////////////////////////////////////////////////////////////////////
//                                     DEC
////////////////////////////////////////////////////////////////////////////////

template <DataType DT>
void dec_templated(
    DECConfig* dec_config,
    torch::Tensor input,
    torch::Tensor output,
    cudaStream_t stream
) {
    auto dec_context = dec_config->dec_context;

    uint32_t height = input.size(2);
    uint32_t width = output.size(2);
    uint32_t num_rows_to_fetch = dec_config->k_chunk * height / dec_config->chunk_size;

    if (num_rows_to_fetch == 0) {
        return;
    }

    auto input_data = (FP_DTYPE(DT)*) input.data_ptr<ATEN_DTYPE(DT)>();
    auto output_data = (FP_DTYPE(DT)*) output.data_ptr<ATEN_DTYPE(DT)>();

    void* kernelArgs[] = {
        (void*)&(dec_config->q_residual),
        (void*)&input_data,
        (void*)&(dec_context->selected_rows_buffer),
        (void*)&(dec_context->selected_activations_buffer),
        (void*)&output_data,
        (void*)&(dec_config->reordered_scales),
        (void*)&(((FP_DTYPE(DT) *)dec_config->thresholds)[0]),
        (void*)&(((FP_DTYPE(DT) *)dec_config->thresholds)[num_rows_to_fetch - 1]),
        (void*)&height,
        (void*)&width,
        (void*)&(dec_config->num_chunks),
        (void*)&(dec_config->chunk_size),
        (void*)&(dec_config->k_chunk),
        (void*)&(dec_config->channels_per_tb),
        (void*)&(dec_config->reduced_shared_memory)
    };

    cudaLaunchCooperativeKernel(
        (void*)fused_dec_kernel<DT>,
        dec_config->grid_dim,
        dec_config->block_dim,
        kernelArgs,
        dec_config->shared_memory_size,
        stream
    );


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA Error in fused_dec_kernel: ", cudaGetErrorString(err));
    }
}

void dec_stream(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    cudaStream_t stream
) {
    // Check that input and output have the same data type
    TORCH_CHECK(input.scalar_type() == output.scalar_type(), "Mismatched data types between input and output tensors. input type: ", input.scalar_type(), ", output type: ", output.scalar_type());

    // Check that both input and output are on GPU
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output tensors must be on GPU.");

    auto dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);

    auto dtype = input.scalar_type();

    // Check that data type of input matches the data type in DECConfig
    TORCH_CHECK(ATEN2DT(dtype) == dec_config->data_type, "Mismatched data types between input tensor and DECConfig. input type: ", DT2STR(ATEN2DT(dtype)), ", DECConfig type: ", DT2STR(dec_config->data_type));

    // Check that the dimensions of input and output are correct
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, hidden_size).");
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, hidden_size).");
    TORCH_CHECK(input.size(0) == 1, "Batch size must be 1 for input tensor.");
    TORCH_CHECK(output.size(0) == 1, "Batch size must be 1 for output tensor.");
    TORCH_CHECK(input.size(1) == 1, "Sequence length must be 1 for input tensor.");
    TORCH_CHECK(output.size(1) == 1, "Sequence length must be 1 for output tensor.");

    // Check that the dimensions of tensors match the DECConfig
    TORCH_CHECK(input.size(2) == dec_config->num_rows, "input tensor size does not match with DECConfig: input size: ", input.size(2), ", DECConfig size: ", dec_config->num_rows);
    TORCH_CHECK(output.size(2) == dec_config->num_cols, "output tensor size does not match with DECConfig: output size: ", output.size(2), ", DECConfig size: ", dec_config->num_cols);

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");

    if (dtype == at::kFloat) {
        dec_templated<DataType::FP32>(dec_config, input, output, stream);
    } else if (dtype == at::kHalf) {
        dec_templated<DataType::FP16>(dec_config, input, output, stream);
    } else if (dtype == at::kBFloat16) {
        dec_templated<DataType::BF16>(dec_config, input, output, stream);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

void dec(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dec_stream(dec_config_uintptr, input, output, stream);
}

////////////////////////////////////////////////////////////////////////////////
//                                Unfused DEC
////////////////////////////////////////////////////////////////////////////////

template<DataType DT>
void unfused_dec_templated(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    cudaStream_t stream
) {
    auto dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);
    auto dec_context = dec_config->dec_context;

    uint32_t height = input.size(2);
    uint32_t width = output.size(2);
    uint32_t num_rows_to_fetch = dec_config->k_chunk * height / dec_config->chunk_size;

    if (num_rows_to_fetch == 0) {
        return;
    }

    auto input_data = (FP_DTYPE(DT)*) input.data_ptr<ATEN_DTYPE(DT)>();
    auto output_data = (FP_DTYPE(DT)*) output.data_ptr<ATEN_DTYPE(DT)>();


    dec_kernel<DT><<<dec_config->grid_dim, dec_config->block_dim, dec_config->shared_memory_size, stream>>>(
        dec_config->q_residual,
        input_data,
        dec_context->selected_rows_buffer,
        output_data,
        (FP_DTYPE(DT)*)dec_config->reordered_scales,
        height,
        width,
        dec_config->num_chunks,
        dec_config->chunk_size,
        dec_config->k_chunk,
        dec_config->channels_per_tb,
        dec_config->reduced_shared_memory
    );


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA Error in dec_kernel: ", cudaGetErrorString(err));
    }
}

void unfused_dec_stream(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    cudaStream_t stream
) {
    // Check that input and output have the same data type
    TORCH_CHECK(input.scalar_type() == output.scalar_type(), "Mismatched data types between input and output tensors. input type: ", input.scalar_type(), ", output type: ", output.scalar_type());

    auto dtype = input.scalar_type();

    // Check that data type of input matches the data type in DECConfig
    auto dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);
    TORCH_CHECK(ATEN2DT(dtype) == dec_config->data_type, "Mismatched data types between input tensor and DECConfig. input type: ", DT2STR(ATEN2DT(dtype)), ", DECConfig type: ", DT2STR(dec_config->data_type));

    // Check that the dimensions of input and output are correct
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, hidden_size).");
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, hidden_size).");
    TORCH_CHECK(input.size(0) == 1, "Batch size must be 1 for input tensor.");
    TORCH_CHECK(output.size(0) == 1, "Batch size must be 1 for output tensor.");
    TORCH_CHECK(input.size(1) == 1, "Sequence length must be 1 for input tensor.");
    TORCH_CHECK(output.size(1) == 1, "Sequence length must be 1 for output tensor.");

    // Check that the dimensions of tensors match the DECConfig
    TORCH_CHECK(input.size(2) == dec_config->num_rows, "input tensor size does not match with DECConfig");
    TORCH_CHECK(output.size(2) == dec_config->num_cols, "output tensor size does not match with DECConfig");

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");

    if (dtype == at::kFloat) {
        unfused_dec_templated<DataType::FP32>(dec_config_uintptr, input, output, stream);
    } else if (dtype == at::kHalf) {
        unfused_dec_templated<DataType::FP16>(dec_config_uintptr, input, output, stream);
    } else if (dtype == at::kBFloat16) {
        unfused_dec_templated<DataType::BF16>(dec_config_uintptr, input, output, stream);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

void unfused_dec(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    unfused_dec_stream(dec_config_uintptr, input, output, stream);
}

////////////////////////////////////////////////////////////////////////////////
//                                     ANYPREC
////////////////////////////////////////////////////////////////////////////////

template<DataType DT>
void anyprec_gemv_templated(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    uint32_t M = input.size(0);
    uint32_t N = output.size(2);
    uint32_t K = input.size(2);

    anyprec_matmul<DT>(
        (FP_DTYPE(DT)*)input.data_ptr<ATEN_DTYPE(DT)>(),
        (FP_DTYPE(DT)*)output.data_ptr<ATEN_DTYPE(DT)>(),
        (uint32_t*)qweight.data_ptr<int>(),
        (FP_DTYPE(DT)*)lut.data_ptr<ATEN_DTYPE(DT)>(),
        M, N, K,
        bitwidth,
        stream
    );
}

void anyprec_gemv_stream(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    TORCH_CHECK(bitwidth >= 3 && bitwidth <= 8, "Bitwidth must be between 3 and 8.");
    TORCH_CHECK(input.scalar_type() == lut.scalar_type() && input.scalar_type() == output.scalar_type(), 
                "Mismatched data types between input, lut, and output tensors.");
    TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight tensor must be of type int.");
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, hidden_size).");
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, hidden_size).");

    // lut is of shape (output_feat, 2 ** bitwidth)
    TORCH_CHECK(lut.dim() == 2 && lut.size(1) == (1 << bitwidth) && lut.size(0) == output.size(2),
    "lut tensor must be of shape (output_feat, 2 ** bitwidth). Expected (", output.size(2), ", ", 1 << bitwidth, "), got (", lut.size(0), ", ", lut.size(1), ").");

    // qweight is of shape (bitwidth, output_feat, input_feat / 32)
    TORCH_CHECK(qweight.dim() == 3 && qweight.size(0) >= bitwidth && qweight.size(2) == input.size(2) / 32 && qweight.size(1) == output.size(2),
    "qweight tensor must be of shape (>=bitwidth, output_feat, input_feat / 32). Expected (>=", bitwidth, ", ", output.size(2), ", ", input.size(2) / 32, "), got (", qweight.size(0), ", ", qweight.size(1), ", ", qweight.size(2), ").");

    // Check that sequence length is 1
    TORCH_CHECK(input.size(1) == 1, "Only sequence length of 1 is supported.");
    TORCH_CHECK(output.size(1) == 1, "Only sequence length of 1 is supported.");

    // Check that input and output are both on GPU
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output tensors must be on GPU.");

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(qweight.is_contiguous(), "qweight tensor must be contiguous.");
    TORCH_CHECK(lut.is_contiguous(), "lut tensor must be contiguous.");

    auto dtype = input.scalar_type();
    if (dtype == at::kFloat) {
        TORCH_CHECK(false, "Any-Precision GEMV does not support float data type. Please use half or bfloat16.");
        //anyprec_gemv_templated<DataType::FP32>(input, output, qweight, lut, bitwidth, stream);
    } else if (dtype == at::kHalf) {
        anyprec_gemv_templated<DataType::FP16>(input, output, qweight, lut, bitwidth, stream);
    } else if (dtype == at::kBFloat16) {
        anyprec_gemv_templated<DataType::BF16>(input, output, qweight, lut, bitwidth, stream);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

void anyprec_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    anyprec_gemv_stream(input, output, qweight, lut, bitwidth, stream);
}

////////////////////////////////////////////////////////////////////////////////
//                               ANYPREC DEQUANT
////////////////////////////////////////////////////////////////////////////////

template<DataType DT>
torch::Tensor anyprec_dequant_templated(
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    assert(qweight.ndimension() == 3 && qweight.dtype() == torch::kInt && (lut.dtype() == torch::kHalf || lut.dtype() == torch::kBFloat16));
    assert(qweight.device() == lut.device() && qweight.is_cuda());
    assert(bitwidth >= 2 && bitwidth <= 8);
    const int N = qweight.size(1);
    const int K = qweight.size(2) * 32;

    auto options = torch::TensorOptions().dtype(lut.dtype()).device(qweight.device());
    at::Tensor weight = torch::empty({N, K}, options);

    anyprec_dequant_kbit<DT>(
        (uint32_t *)qweight.data_ptr<int>(),
        N, K,
        (FP_DTYPE(DT) *)lut.data_ptr<ATEN_DTYPE(DT)>(),
        (FP_DTYPE(DT) *)weight.data_ptr<ATEN_DTYPE(DT)>(),
        bitwidth,
        stream
    );

    return weight;
}

torch::Tensor anyprec_dequant_stream(
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    auto dtype = lut.scalar_type();
    if (dtype == at::kFloat) {
        TORCH_CHECK(false, "Any-Precision Dequantization does not support float data type. Please use half or bfloat16.");
        //return anyprec_dequant_templated<DataType::FP32>(qweight, lut, bitwidth, stream);
    } else if (dtype == at::kHalf) {
        return anyprec_dequant_templated<DataType::FP16>(qweight, lut, bitwidth, stream);
    } else if (dtype == at::kBFloat16) {
        return anyprec_dequant_templated<DataType::BF16>(qweight, lut, bitwidth, stream);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

torch::Tensor anyprec_dequant(
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return anyprec_dequant_stream(qweight, lut, bitwidth, stream);
}

////////////////////////////////////////////////////////////////////////////////
//                                     LUTGEMM
////////////////////////////////////////////////////////////////////////////////

template<DataType DT>
void lutgemm_gemv_templated(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size,
    cudaStream_t stream
) {
    uint32_t kSize = input.size(2);
    uint32_t mSize = output.size(2);

    dim3 grid((mSize + M_TILE_SIZE - 1) / M_TILE_SIZE,
              (kSize + K_TILE_SIZE - 1) / K_TILE_SIZE);
    dim3 block(NUM_THREADS);

    nqmv_bias<DT><<<grid, block, 0, stream>>>(
        (uint32_t*) q_weight.data_ptr<int32_t>(),
        (FP_DTYPE(DT)*) alpha.data_ptr<ATEN_DTYPE(DT)>(),
        (FP_DTYPE(DT)*) q_bias.data_ptr<ATEN_DTYPE(DT)>(),
        (FP_DTYPE(DT)*) input.data_ptr<ATEN_DTYPE(DT)>(),
        (FP_DTYPE(DT)*) output.data_ptr<ATEN_DTYPE(DT)>(),
        mSize, kSize, bitwidth, group_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
}

void lutgemm_gemv_stream(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size,
    cudaStream_t stream
) {
    TORCH_CHECK(bitwidth >= 1 && bitwidth <= 8, "Bitwidth must be between 1 and 8.");
    TORCH_CHECK(input.scalar_type() == alpha.scalar_type() && input.scalar_type() == q_bias.scalar_type() && input.scalar_type() == output.scalar_type(), "Mismatched data types between input, alpha, q_bias, and output tensors.");
    // Check that input is of shape (batch_size, seq_len, input_feat)
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, input_feat).");
    // Check that output is of shape (batch_size, seq_len, output_feat)
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, output_feat).");

    // Only allow single batch size and sequence length
    TORCH_CHECK(input.size(0) == 1, "Batch size must be 1 for input tensor.");
    TORCH_CHECK(input.size(1) == 1, "Sequence length must be 1 for input tensor.");
    TORCH_CHECK(output.size(0) == 1, "Batch size must be 1 for output tensor.");
    TORCH_CHECK(output.size(1) == 1, "Sequence length must be 1 for output tensor.");

    // Check that input and output are both on GPU
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output tensors must be on GPU.");

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(q_weight.is_contiguous(), "q_weight tensor must be contiguous.");
    TORCH_CHECK(alpha.is_contiguous(), "alpha tensor must be contiguous.");
    TORCH_CHECK(q_bias.is_contiguous(), "q_bias tensor must be contiguous.");

    uint32_t kSize = input.size(2);
    uint32_t mSize = output.size(2);
    uint32_t num_groups = kSize / group_size;

    // check that q_weight is of shape (input_feat / 32, bitwidth, output_feat)
    TORCH_CHECK(q_weight.dim() == 3 && q_weight.size(0) == kSize / 32 && q_weight.size(1) == bitwidth && q_weight.size(2) == mSize, "q_weight tensor must be of shape (input_feat / 32, bitwidth, output_feat). Expected (", kSize / 32, ", ", bitwidth, ", ", mSize, "), got (", q_weight.size(0), ", ", q_weight.size(1), ", ", q_weight.size(2), ").");
    // check that alpha is of shape (num_groups, bitwidth, mSize)
    TORCH_CHECK(alpha.dim() == 3 && alpha.size(0) == num_groups && alpha.size(1) == bitwidth && alpha.size(2) == mSize, 
                "alpha tensor must be of shape (num_groups, bitwidth, output_feat). Expected (", num_groups, ", ", bitwidth, ", ", mSize, "), got (", alpha.size(0), ", ", alpha.size(1), ", ", alpha.size(2), ").");

    auto dtype = input.scalar_type();
    if (dtype == at::kFloat) {
        lutgemm_gemv_templated<DataType::FP32>(input, output, q_weight, alpha, q_bias, bitwidth, group_size, stream);
    } else if (dtype == at::kHalf) {
        lutgemm_gemv_templated<DataType::FP16>(input, output, q_weight, alpha, q_bias, bitwidth, group_size, stream);
    } else if (dtype == at::kBFloat16) {
        lutgemm_gemv_templated<DataType::BF16>(input, output, q_weight, alpha, q_bias, bitwidth, group_size, stream);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

void lutgemm_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lutgemm_gemv_stream(input, output, q_weight, alpha, q_bias, bitwidth, group_size, stream);
}

////////////////////////////////////////////////////////////////////////////////
//                                     SQLLM
////////////////////////////////////////////////////////////////////////////////

template<DataType DT>
void sqllm_gemv_templated(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    uint32_t height = input.size(2);
    uint32_t width = qweight.size(1);
    uint32_t matrix_height = height / 32 * bitwidth;

    dim3 threads(SQLLM_BLOCKWIDTH);
    dim3 grid = dim3(
        (matrix_height + (bitwidth == 3 ? SQLLM_BLOCKHEIGHT3 : SQLLM_BLOCKHEIGHT4) - 1) / 
        (bitwidth == 3 ? SQLLM_BLOCKHEIGHT3 : SQLLM_BLOCKHEIGHT4),
        (width + SQLLM_BLOCKWIDTH - 1) / SQLLM_BLOCKWIDTH
    );

    if (bitwidth == 3) {
        VecQuant3MatMulKernelNUQPerChannel<DT><<<grid, threads, 0, stream>>>(
            (FP_DTYPE(DT)*)input.data_ptr<ATEN_DTYPE(DT)>(),
            (int*)qweight.data_ptr<int>(),
            (FP_DTYPE(DT)*)output.data_ptr<ATEN_DTYPE(DT)>(),
            (FP_DTYPE(DT)*)lut.data_ptr<ATEN_DTYPE(DT)>(),
            static_cast<int>(height), static_cast<int>(width)
        );
    } else if (bitwidth == 4) {
        VecQuant4MatMulKernelNUQPerChannel<DT><<<grid, threads, 0, stream>>>(
            (FP_DTYPE(DT)*)input.data_ptr<ATEN_DTYPE(DT)>(),
            (int*)qweight.data_ptr<int>(),
            (FP_DTYPE(DT)*)output.data_ptr<ATEN_DTYPE(DT)>(),
            (FP_DTYPE(DT)*)lut.data_ptr<ATEN_DTYPE(DT)>(),
            static_cast<int>(height), static_cast<int>(width)
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in sqllm_gemv: ", cudaGetErrorString(err));
}

void sqllm_gemv_stream(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    TORCH_CHECK(bitwidth == 3 || bitwidth == 4, "Bitwidth must be 3 or 4.");
    TORCH_CHECK(input.scalar_type() == lut.scalar_type() && input.scalar_type() == output.scalar_type(), 
                "Mismatched data types between input, lut, and output tensors.");
    TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight tensor must be of type int.");
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, hidden_size).");
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, hidden_size).");

    // Only allow single batch size and sequence length
    TORCH_CHECK(input.size(0) == 1, "Batch size must be 1 for input tensor.");
    TORCH_CHECK(input.size(1) == 1, "Sequence length must be 1 for input tensor.");
    TORCH_CHECK(output.size(0) == 1, "Batch size must be 1 for output tensor.");
    TORCH_CHECK(output.size(1) == 1, "Sequence length must be 1 for output tensor.");

    // Check that input and output are both on GPU
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output tensors must be on GPU.");

    // Check that lut is of shape (output_feat, 2 ** bitwidth)
    TORCH_CHECK(lut.dim() == 2 && lut.size(1) == (1 << bitwidth) && lut.size(0) == output.size(2),
    "lut tensor must be of shape (output_feat, 2 ** bitwidth). Expected (", output.size(2), ", ", 1 << bitwidth, "), got (", lut.size(0), ", ", lut.size(1), ").");

    // Check that qweight is of shape (input_feat * bitwidth / 32, output_feat)
    TORCH_CHECK(qweight.dim() == 2 && qweight.size(1) == lut.size(0) && qweight.size(0) == input.size(2) * bitwidth / 32,
    "qweight tensor must be of shape (input_feat * bitwidth / 32, output_feat). Expected (", input.size(2) * bitwidth / 32, ", ", lut.size(0), "), got (", qweight.size(0), ", ", qweight.size(1), ").");

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(qweight.is_contiguous(), "qweight tensor must be contiguous.");
    TORCH_CHECK(lut.is_contiguous(), "lut tensor must be contiguous.");

    auto dtype = input.scalar_type();
    if (dtype == at::kFloat) {
        sqllm_gemv_templated<DataType::FP32>(input, output, qweight, lut, bitwidth, stream);
    } else if (dtype == at::kHalf) {
        sqllm_gemv_templated<DataType::FP16>(input, output, qweight, lut, bitwidth, stream);
    } else if (dtype == at::kBFloat16) {
        sqllm_gemv_templated<DataType::BF16>(input, output, qweight, lut, bitwidth, stream);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

void sqllm_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sqllm_gemv_stream(input, output, qweight, lut, bitwidth, stream);
}

////////////////////////////////////////////////////////////////////////////////
//                             DecDEC + ANYPREC
////////////////////////////////////////////////////////////////////////////////

template<DataType DT>
void dec_anyprec_templated(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor lut,
    int bitwidth
) {
    auto dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);
    auto dec_context = dec_config->dec_context;

    cudaStream_t main_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t sub_stream = dec_context->sub_stream;
    cudaEvent_t event_1 = dec_config->event_1;
    cudaEvent_t event_2 = dec_config->event_2;

    // Create event on main stream
    cudaEventRecord(event_1, main_stream);

    // Launch DEC kernel on main stream
    dec_stream(dec_config_uintptr, input, output, main_stream);

    // Wait for event_1 on sub_stream before launching GEMV
    cudaStreamWaitEvent(sub_stream, event_1, 0);

    // Launch GEMV kernel with anyprec_gemv
    anyprec_gemv_stream(input, output, q_weight, lut, bitwidth, sub_stream);

    // Sync main stream with sub stream
    cudaEventRecord(event_2, sub_stream);
    cudaStreamWaitEvent(main_stream, event_2, 0);
}

void dec_anyprec(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor lut,
    int bitwidth
) {
    auto dtype = input.scalar_type();
    if (dtype == at::kHalf) {
        dec_anyprec_templated<DataType::FP16>(dec_config_uintptr, input, output, q_weight, lut, bitwidth);
    } else if (dtype == at::kBFloat16) {
        dec_anyprec_templated<DataType::BF16>(dec_config_uintptr, input, output, q_weight, lut, bitwidth);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

////////////////////////////////////////////////////////////////////////////////
//                             DecDEC + LUTGEMM
////////////////////////////////////////////////////////////////////////////////

template<DataType DT>
void dec_lutgemm_templated(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size
) {
    auto dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);
    auto dec_context = dec_config->dec_context;

    cudaStream_t main_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t sub_stream = dec_context->sub_stream;
    cudaEvent_t event_1 = dec_config->event_1;
    cudaEvent_t event_2 = dec_config->event_2;

    // Create event on main stream
    cudaEventRecord(event_1, main_stream);

    // Launch DEC kernel on main stream
    dec_stream(dec_config_uintptr, input, output, main_stream);

    // Wait for event_1 on sub_stream before launching GEMV
    cudaStreamWaitEvent(sub_stream, event_1, 0);

    // Launch GEMV kernel with lutgemm_gemv
    lutgemm_gemv_stream(input, output, q_weight, alpha, q_bias, bitwidth, group_size, sub_stream);

    // Sync main stream with sub stream
    cudaEventRecord(event_2, sub_stream);
    cudaStreamWaitEvent(main_stream, event_2, 0);
}

void dec_lutgemm(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size
) {
    auto dtype = input.scalar_type();
    if (dtype == at::kFloat) {
        dec_lutgemm_templated<DataType::FP32>(dec_config_uintptr, input, output, q_weight, alpha, q_bias, bitwidth, group_size);
    } else if (dtype == at::kHalf) {
        dec_lutgemm_templated<DataType::FP16>(dec_config_uintptr, input, output, q_weight, alpha, q_bias, bitwidth, group_size);
    } else if (dtype == at::kBFloat16) {
        dec_lutgemm_templated<DataType::BF16>(dec_config_uintptr, input, output, q_weight, alpha, q_bias, bitwidth, group_size);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

////////////////////////////////////////////////////////////////////////////////
//                             DecDEC + SQLLM
////////////////////////////////////////////////////////////////////////////////

template<DataType DT>
void dec_sqllm_templated(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor lut,
    int bitwidth
) {
    auto dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);
    auto dec_context = dec_config->dec_context;

    cudaStream_t main_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t sub_stream = dec_context->sub_stream;
    cudaEvent_t event_1 = dec_config->event_1;
    cudaEvent_t event_2 = dec_config->event_2;

    // Create event on main stream
    cudaEventRecord(event_1, main_stream);

    // Launch DEC kernel on main stream
    dec_stream(dec_config_uintptr, input, output, main_stream);

    // Wait for event_1 on sub_stream before launching GEMV
    cudaStreamWaitEvent(sub_stream, event_1, 0);

    // Launch GEMV kernel with sqllm_gemv
    sqllm_gemv_stream(input, output, q_weight, lut, bitwidth, sub_stream);

    // Sync main stream with sub stream
    cudaEventRecord(event_2, sub_stream);
    cudaStreamWaitEvent(main_stream, event_2, 0);
}

void dec_sqllm(
    uintptr_t dec_config_uintptr,
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor lut,
    int bitwidth
) {
    auto dtype = input.scalar_type();
    if (dtype == at::kFloat) {
        dec_sqllm_templated<DataType::FP32>(dec_config_uintptr, input, output, q_weight, lut, bitwidth);
    } else if (dtype == at::kHalf) {
        dec_sqllm_templated<DataType::FP16>(dec_config_uintptr, input, output, q_weight, lut, bitwidth);
    } else if (dtype == at::kBFloat16) {
        dec_sqllm_templated<DataType::BF16>(dec_config_uintptr, input, output, q_weight, lut, bitwidth);
    } else {
        TORCH_CHECK(false, "Unsupported data type.");
    }
}

////////////////////////////////////////////////////////////////////////////////
//                      SM‑burning dummy helpers (no memory)                 //
////////////////////////////////////////////////////////////////////////////////
/*
 * A pure‑compute kernel that keeps one block resident per SM while touching
 * almost no global/shared memory. Used to occupy SMs via long compute.
 */
__device__ float sm_burner_sink[1];

__global__ void sm_burner_kernel(long long iters)
{
    __shared__ float sdata[1024];  // 4 KB of shared memory
    sdata[threadIdx.x] = threadIdx.x;

    // Register-heavy accumulation
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        acc[i] = threadIdx.x + i;
    }

    // Compute loop
    for (long long i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            acc[j] = acc[j] * 1.00001f + 0.0001f;
        }
    }

    // Write one float per thread to shared memory
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        sum += acc[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Atomic add (only from one thread) to prevent optimization
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(sm_burner_sink, sdata[0]);
    }
}

static inline void launch_sm_burner(int dummy_sm,
                                    long long dummy_iters,
                                    cudaStream_t stream)
{
    if (dummy_sm <= 0) return;

    int threads_per_block = 1024;
    sm_burner_kernel<<<dummy_sm, threads_per_block, 0, stream>>>(dummy_iters);
}

////////////////////////////////////////////////////////////////////////////////
//            Dummy versions that REPLACE the DEC stage with a burner         //
////////////////////////////////////////////////////////////////////////////////
/*
  * Use to test the performance of the GEMV kernels when SMs are limited.
 */
template <DataType DT>
static inline void dummy_anyprec_templated(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    int dummy_sm,
    long long dummy_iters
) {
    cudaStream_t main_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t sub_stream;
    cudaStreamCreateWithFlags(&sub_stream, cudaStreamNonBlocking);

    cudaEvent_t event_1, event_2;
    cudaEventCreate(&event_1);
    cudaEventCreate(&event_2);

    // Record event on main stream before dummy kernel
    cudaEventRecord(event_1, main_stream);

    // Launch dummy kernel on main stream (replacing DEC)
    launch_sm_burner(dummy_sm, dummy_iters, main_stream);

    cudaStreamWaitEvent(sub_stream, event_1, 0);

    // Launch GEMV kernel in sub stream
    anyprec_gemv_stream(input, output, qweight, lut, bitwidth, sub_stream);
    // Ensure sub stream finishes before main continues
    cudaEventRecord(event_2, sub_stream);
    cudaStreamWaitEvent(main_stream, event_2, 0);
}

void dummy_anyprec(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    int dummy_sm,
    long long dummy_iters
) {
    auto dtype = input.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
                "dummy_anyprec supports FP16/BF16 only.");
    if (dtype == at::kHalf) {
        dummy_anyprec_templated<DataType::FP16>(input, output, qweight, lut,
                                                bitwidth, dummy_sm, dummy_iters);
    } else {
        dummy_anyprec_templated<DataType::BF16>(input, output, qweight, lut,
                                                bitwidth, dummy_sm, dummy_iters);
    }
}

////////////////////////////////////////////////////////////////////////////////

template <DataType DT>
static inline void dummy_lutgemm_templated(
    torch::Tensor  input,
    torch::Tensor  output,
    torch::Tensor  qweight,
    torch::Tensor  alpha,
    torch::Tensor  qbias,
    int bitwidth,
    int group_size,
    int dummy_sm,
    long long dummy_iters
) {
    cudaStream_t main_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t sub_stream;
    cudaStreamCreateWithFlags(&sub_stream, cudaStreamNonBlocking);

    cudaEvent_t event_1, event_2;
    cudaEventCreate(&event_1);
    cudaEventCreate(&event_2);

    cudaEventRecord(event_1, main_stream);
    
    launch_sm_burner(dummy_sm, dummy_iters, main_stream);

    cudaStreamWaitEvent(sub_stream, event_1, 0);

    lutgemm_gemv_stream(input, output, qweight, alpha, qbias,
                        bitwidth, group_size, sub_stream);

    cudaEventRecord(event_2, sub_stream);
    cudaStreamWaitEvent(main_stream, event_2, 0);
}

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
) {
    auto dtype = input.scalar_type();
    if (dtype == at::kFloat)
        dummy_lutgemm_templated<DataType::FP32>(input, output, qweight, alpha,
                                                qbias, bitwidth, group_size,
                                                dummy_sm, dummy_iters);
    else if (dtype == at::kHalf)
        dummy_lutgemm_templated<DataType::FP16>(input, output, qweight, alpha,
                                                qbias, bitwidth, group_size,
                                                dummy_sm, dummy_iters);
    else if (dtype == at::kBFloat16)
        dummy_lutgemm_templated<DataType::BF16>(input, output, qweight, alpha,
                                                qbias, bitwidth, group_size,
                                                dummy_sm, dummy_iters);
    else
        TORCH_CHECK(false, "Unsupported dtype for dummy_lutgemm");
}

////////////////////////////////////////////////////////////////////////////////

template <DataType DT>
static inline void dummy_sqllm_templated(
    torch::Tensor  input,
    torch::Tensor  output,
    torch::Tensor  qweight,
    torch::Tensor  lut,
    int bitwidth,
    int dummy_sm,
    long long dummy_iters
) {
    cudaStream_t main_stream = at::cuda::getCurrentCUDAStream();
    cudaStream_t sub_stream;
    cudaStreamCreateWithFlags(&sub_stream, cudaStreamNonBlocking);

    cudaEvent_t event_1, event_2;
    cudaEventCreate(&event_1);
    cudaEventCreate(&event_2);

    cudaEventRecord(event_1, main_stream);

    launch_sm_burner(dummy_sm, dummy_iters, main_stream);

    cudaStreamWaitEvent(sub_stream, event_1, 0);

    sqllm_gemv_stream(input, output, qweight, lut, bitwidth, sub_stream);

    cudaEventRecord(event_2, sub_stream);
    cudaStreamWaitEvent(main_stream, event_2, 0);
}

void dummy_sqllm(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    int dummy_sm,
    long long dummy_iters
) {
    auto dtype = input.scalar_type();
    TORCH_CHECK(bitwidth == 3 || bitwidth == 4,
                "SQLLM bitwidth must be 3 or 4.");
    if (dtype == at::kFloat)
        dummy_sqllm_templated<DataType::FP32>(input, output, qweight, lut,
                                              bitwidth, dummy_sm, dummy_iters);
    else if (dtype == at::kHalf)
        dummy_sqllm_templated<DataType::FP16>(input, output, qweight, lut,
                                              bitwidth, dummy_sm, dummy_iters);
    else if (dtype == at::kBFloat16)
        dummy_sqllm_templated<DataType::BF16>(input, output, qweight, lut,
                                              bitwidth, dummy_sm, dummy_iters);
    else
        TORCH_CHECK(false, "Unsupported dtype for dummy_sqllm");
}
