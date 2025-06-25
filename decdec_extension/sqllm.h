#ifndef SQLLM_CUH
#define SQLLM_CUH

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
#include "datatype.h"
#include "typetraits.h"

#include <torch/extension.h>
#include <cuda_runtime.h>

template<DataType DT>
__global__ void VecQuant4MatMulKernelNUQPerChannel(
        const FP_DTYPE(DT) *__restrict__ vec,
        const int *__restrict__ mat,
        FP_DTYPE(DT) *__restrict__ mul,
        const FP_DTYPE(DT) *__restrict__ lookup_table,
        int height,
        int width
);

template<DataType DT>
__global__ void VecQuant3MatMulKernelNUQPerChannel(
        const FP_DTYPE(DT) *__restrict__ vec,
        const int *__restrict__ mat,
        FP_DTYPE(DT) *__restrict__ mul,
        const FP_DTYPE(DT) *__restrict__ lookup_table,
        int height,
        int width
);

#endif

