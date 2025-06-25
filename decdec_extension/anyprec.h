#ifndef ANYPREC_CUH
#define ANYPREC_CUH

#include <cstdint>
#include "datatype.h"
#include "typetraits.h"

template<DataType DT>
void anyprec_matmul(
        FP_DTYPE(DT) *in,            // [M, K]
        FP_DTYPE(DT) *out,           // [M, N]
        uint32_t *qweight,       // [w_bits, N, K/32]
        FP_DTYPE(DT) *lut,           // [out_size, num_centroids]
        uint32_t M,              // batch size
        uint32_t N,              // output size
        uint32_t K,              // input size
        int w_bits,               // weight bits
        cudaStream_t stream
);


template<DataType DT>
void anyprec_dequant_kbit(
    const uint32_t *qweight,
    const uint32_t N, const uint32_t K,
    const FP_DTYPE(DT) *lut, FP_DTYPE(DT) *weight,
    int w_bits,
    cudaStream_t stream
);

#endif // ANYPREC_CUH