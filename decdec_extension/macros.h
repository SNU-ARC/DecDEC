#ifndef MACROS_CUH
#define MACROS_CUH

#define DEBUG 0

#define LUT_GEMM_GROUP_SIZE 128

#define CHUNK_SIZE 1024

#define BIT_WIDTH 4
#define THREADS_PER_BLOCK 1024

#define WARP_SIZE 32
#define MAX_MATRIX_COLS_PER_BLOCK 256  // this value must be limited for the sake of shared memory

#define NIBBLES_PER_ELEMENT (32 / BIT_WIDTH)  // how many residual values to pack per matrix element

#define NUM_THRESHOLDS 32  // Power of 2 between 1 and 32

#if DEBUG
    #define ASSERT(x) assert(x)
#else
    #define ASSERT(x)
#endif

template<typename T>
inline bool fp_compare(T a, T b) {
    return a > b;
}

#endif // MACROS_CUH
