#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "macros.h"
#include "sqllm.h"
#include "typetraits.h"
#include "datatype.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>

#define SQLLM_BLOCKWIDTH 128
#define SQLLM_BLOCKHEIGHT3 12
#define SQLLM_BLOCKHEIGHT4 16

__device__ inline unsigned int as_unsigned(int i) {
    return *reinterpret_cast < unsigned int * > ( &i);
}


template<DataType DT>
__global__ void VecQuant3MatMulKernelNUQPerChannel(
        const FP_DTYPE(DT) *__restrict__ vec,
        const int *__restrict__ mat,
        FP_DTYPE(DT) *__restrict__ mul,
        const FP_DTYPE(DT) *__restrict__ lookup_table,
        int height,
        int width
) {

    int row = SQLLM_BLOCKHEIGHT3 * blockIdx.x;
    int col = SQLLM_BLOCKWIDTH * blockIdx.y + threadIdx.x;

    __shared__ FP_DTYPE(DT) blockvec[SQLLM_BLOCKWIDTH];
    blockvec[threadIdx.x] = vec[(row / SQLLM_BLOCKHEIGHT3) * SQLLM_BLOCKWIDTH + threadIdx.x];

    //Modified dequant block
    __shared__ FP_DTYPE(DT) deq2[8][SQLLM_BLOCKWIDTH];
    int off = threadIdx.x;
    int column_offset = col * 8;
    for (int val = 0; val < 8; val += 1) {
        int lut_index = column_offset + val;
        deq2[val][off] = lookup_table[lut_index];
    }

    int i = width * row + col;
    int k = 0;

    FP_DTYPE(DT) res = 0;

    unsigned int tmp1;
    unsigned int tmp2;
    unsigned int tmp;

    __syncthreads();

    while (k < SQLLM_BLOCKWIDTH) {
        tmp1 = as_unsigned(mat[i]);

        res += deq2[(tmp1 >> 0) & 0x7][off] * blockvec[k + 0];
        res += deq2[(tmp1 >> 3) & 0x7][off] * blockvec[k + 1];
        res += deq2[(tmp1 >> 6) & 0x7][off] * blockvec[k + 2];
        res += deq2[(tmp1 >> 9) & 0x7][off] * blockvec[k + 3];
        res += deq2[(tmp1 >> 12) & 0x7][off] * blockvec[k + 4];
        res += deq2[(tmp1 >> 15) & 0x7][off] * blockvec[k + 5];
        res += deq2[(tmp1 >> 18) & 0x7][off] * blockvec[k + 6];
        res += deq2[(tmp1 >> 21) & 0x7][off] * blockvec[k + 7];
        res += deq2[(tmp1 >> 24) & 0x7][off] * blockvec[k + 8];
        res += deq2[(tmp1 >> 27) & 0x7][off] * blockvec[k + 9];

        i += width;
        tmp2 = as_unsigned(mat[i]);
        tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
        tmp2 >>= 1;
        res += deq2[(tmp >> 0) & 0x7][off] * blockvec[k + 10];
        k += 11;
        res += deq2[(tmp2 >> 0) & 0x7][off] * blockvec[k + 0];
        res += deq2[(tmp2 >> 3) & 0x7][off] * blockvec[k + 1];
        res += deq2[(tmp2 >> 6) & 0x7][off] * blockvec[k + 2];
        res += deq2[(tmp2 >> 9) & 0x7][off] * blockvec[k + 3];
        res += deq2[(tmp2 >> 12) & 0x7][off] * blockvec[k + 4];
        res += deq2[(tmp2 >> 15) & 0x7][off] * blockvec[k + 5];
        res += deq2[(tmp2 >> 18) & 0x7][off] * blockvec[k + 6];
        res += deq2[(tmp2 >> 21) & 0x7][off] * blockvec[k + 7];
        res += deq2[(tmp2 >> 24) & 0x7][off] * blockvec[k + 8];
        res += deq2[(tmp2 >> 27) & 0x7][off] * blockvec[k + 9];

        i += width;
        tmp1 = as_unsigned(mat[i]);
        tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
        tmp1 >>= 2;
        res += deq2[(tmp >> 0) & 0x7][off] * blockvec[k + 10];
        k += 11;
        res += deq2[(tmp1 >> 0) & 0x7][off] * blockvec[k + 0];
        res += deq2[(tmp1 >> 3) & 0x7][off] * blockvec[k + 1];
        res += deq2[(tmp1 >> 6) & 0x7][off] * blockvec[k + 2];
        res += deq2[(tmp1 >> 9) & 0x7][off] * blockvec[k + 3];
        res += deq2[(tmp1 >> 12) & 0x7][off] * blockvec[k + 4];
        res += deq2[(tmp1 >> 15) & 0x7][off] * blockvec[k + 5];
        res += deq2[(tmp1 >> 18) & 0x7][off] * blockvec[k + 6];
        res += deq2[(tmp1 >> 21) & 0x7][off] * blockvec[k + 7];
        res += deq2[(tmp1 >> 24) & 0x7][off] * blockvec[k + 8];
        res += deq2[(tmp1 >> 27) & 0x7][off] * blockvec[k + 9];
        i += width;
        k += 10;
    }

    ATOMIC_ADD(DT, &mul[col], res);
}


template<DataType DT>
__global__ void VecQuant4MatMulKernelNUQPerChannel(
        const FP_DTYPE(DT) *__restrict__ vec,
        const int *__restrict__ mat,
        FP_DTYPE(DT) *__restrict__ mul,
        const FP_DTYPE(DT) *__restrict__ lookup_table,
        int height,
        int width
) {

    int row = SQLLM_BLOCKHEIGHT4 * blockIdx.x;
    int col = SQLLM_BLOCKWIDTH * blockIdx.y + threadIdx.x;

    __shared__ FP_DTYPE(DT) blockvec[SQLLM_BLOCKWIDTH];
    blockvec[threadIdx.x] = vec[(row / SQLLM_BLOCKHEIGHT4) * SQLLM_BLOCKWIDTH + threadIdx.x];

    //Modified dequant block
    __shared__ FP_DTYPE(DT) deq2[16][SQLLM_BLOCKWIDTH];
    int off = threadIdx.x;
    int column_offset = col * 16;
    for (int val = 0; val < 16; val += 1) {
        int lut_index = column_offset + val;
        deq2[val][off] = lookup_table[lut_index];
    }

    __syncthreads();

    FP_DTYPE(DT) res = 0;
    int i = width * row + col;
    int k = 0;

    unsigned int tmp;

    while (k < SQLLM_BLOCKWIDTH) {
        tmp = as_unsigned(mat[i]);

        res += deq2[(tmp >> 0) & 0xf][off] * blockvec[k + 0];
        res += deq2[(tmp >> 4) & 0xf][off] * blockvec[k + 1];
        res += deq2[(tmp >> 8) & 0xf][off] * blockvec[k + 2];
        res += deq2[(tmp >> 12) & 0xf][off] * blockvec[k + 3];
        res += deq2[(tmp >> 16) & 0xf][off] * blockvec[k + 4];
        res += deq2[(tmp >> 20) & 0xf][off] * blockvec[k + 5];
        res += deq2[(tmp >> 24) & 0xf][off] * blockvec[k + 6];
        res += deq2[(tmp >> 28) & 0xf][off] * blockvec[k + 7];

        i += width;
        k += 8;
    }

    ATOMIC_ADD(DT, &mul[col], res);
}


// Explicit template instantiation
#define INSTANTIATE_FOR_DATATYPE(DT) \
    template __global__ void VecQuant3MatMulKernelNUQPerChannel<DT>( \
        const FP_DTYPE(DT) *__restrict__ vec, \
        const int *__restrict__ mat, \
        FP_DTYPE(DT) *__restrict__ mul, \
        const FP_DTYPE(DT) *__restrict__ lookup_table, \
        int height, \
        int width \
    ); \
    template __global__ void VecQuant4MatMulKernelNUQPerChannel<DT>( \
        const FP_DTYPE(DT) *__restrict__ vec, \
        const int *__restrict__ mat, \
        FP_DTYPE(DT) *__restrict__ mul, \
        const FP_DTYPE(DT) *__restrict__ lookup_table, \
        int height, \
        int width \
    );

INSTANTIATE_FOR_DATATYPE(DataType::FP32)
INSTANTIATE_FOR_DATATYPE(DataType::FP16)
INSTANTIATE_FOR_DATATYPE(DataType::BF16)