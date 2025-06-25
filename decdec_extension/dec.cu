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
#include "dec.cuh"
#include "typetraits.h"
#include "datatype.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>


#include <assert.h>


template<DataType DT>
__global__ void fused_dec_kernel(
        const uint32_t *__restrict__ cheatsheet, const FP_DTYPE(DT) *__restrict__ activations,
        uint32_t *__restrict__ selected_rows, FP_DTYPE(DT) *__restrict__ selected_activations,
        FP_DTYPE(DT) *__restrict__ final_sum, const FP_DTYPE(DT) *__restrict__ reordered_scales,
        const FP_DTYPE(DT) threshold_a, const FP_DTYPE(DT) threshold_b, const uint32_t num_rows, const uint32_t num_cols,
        const uint32_t num_chunks,
        const uint32_t chunk_size, const uint32_t k_chunk,
        const uint32_t channels_per_tb,
        const bool reduced_shared_memory
) {
    if (k_chunk == 0) {
        return;
    }

    ASSERT(blockDim.x <= channels_per_tb);
    ASSERT(channels_per_tb % blockDim.x == 0);
    ASSERT(blockDim.x % WARP_SIZE == 0); // For the row-selection to work

    const cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    // Handle num_rows_to_fetch, including the case where num_rows is not a multiple of chunk_size
    uint32_t num_rows_to_fetch = k_chunk * num_rows / chunk_size;

    // Kernel Launch Overhead: 2 us
    ///////////////////////////////// Shared Memory Initialization /////////////////////////////////
    __shared__ uint32_t shared_counts[NUM_THRESHOLDS];
    extern __shared__ unsigned char shared_memory[];
    auto (*selected_rows_by_threshold)[NUM_THRESHOLDS] = reinterpret_cast<uint32_t (*)[NUM_THRESHOLDS]>(shared_memory);
    auto *shared_activations = reinterpret_cast<FP_DTYPE(DT) *>(&selected_rows_by_threshold[k_chunk]);

    // Shared memory can be safely reused for the fetch phase
    uint32_t *shared_selected_rows;
    FP_DTYPE(DT) *shared_selected_activations, *unpacked, *shared_reordered_scales;
    if (!reduced_shared_memory) {
        // this must come first, for alignment
        shared_selected_rows = reinterpret_cast<uint32_t *>(&shared_memory);
        unpacked = reinterpret_cast<FP_DTYPE(DT) *>(&shared_selected_rows[num_rows_to_fetch]);
        shared_reordered_scales = reinterpret_cast<FP_DTYPE(DT) *>(&unpacked[blockDim.y * blockDim.x *
                                                                         NIBBLES_PER_ELEMENT]);
        shared_selected_activations = reinterpret_cast<FP_DTYPE(DT) *>(&shared_reordered_scales[blockDim.x *
                                                                                            NIBBLES_PER_ELEMENT]);

    } else {
        // don't use shared memory for selected rows and activations
        unpacked = reinterpret_cast<FP_DTYPE(DT) *>(&shared_memory);
        shared_reordered_scales = reinterpret_cast<FP_DTYPE(DT) *>(&unpacked[blockDim.y * blockDim.x *
                                                                         NIBBLES_PER_ELEMENT]);
    }

    ///////////////////////////////// Fused Row Select /////////////////////////////////
    // -------------------------- Initialization --------------------------
    for (uint32_t chunk_idx = blockIdx.x; chunk_idx < num_chunks; chunk_idx += gridDim.x) {
        // -------------------------- Initialization --------------------------
        const uint32_t chunk_start_idx = chunk_idx * chunk_size;
        const uint32_t chunk_end_idx = min((chunk_idx + 1) * chunk_size, num_rows);

        // Chunk size can be smaller than chunk_size for the last chunk
        const uint32_t clipped_chunk_size = chunk_end_idx - chunk_start_idx;
        const uint32_t adjusted_k_chunk = k_chunk * clipped_chunk_size /
                                                              chunk_size;

        const uint32_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

        const uint32_t warp_idx = thread_idx / WARP_SIZE;
        const uint32_t warp_lane = thread_idx % WARP_SIZE;
        constexpr uint32_t values_per_threshold_per_warp = WARP_SIZE / NUM_THRESHOLDS;
        const uint32_t threshold_idx = warp_lane / values_per_threshold_per_warp;
        const uint32_t row_offset_start = warp_idx * values_per_threshold_per_warp;
        const uint32_t idx_increment = blockDim.x * blockDim.y / NUM_THRESHOLDS;


        // -------------------------- Initialize Shared Memory --------------------------

        // Initialize shared memory counts for each threshold
        if (threadIdx.y == 0 && threadIdx.x < NUM_THRESHOLDS) {
            shared_counts[threadIdx.x] = 0;
        }

        // Loop to handle chunks larger than the block size
        for (uint32_t row_offset = threadIdx.y * blockDim.x + threadIdx.x;
             row_offset < clipped_chunk_size; row_offset += blockDim.x * blockDim.y) {
            const uint32_t row_idx = chunk_start_idx + row_offset;
            if (row_idx < chunk_end_idx) {
                shared_activations[row_offset] = activations[row_idx];
            }
        }
        __syncthreads();

        // -------------------------- Threshold Initialization --------------------------

        FP_DTYPE(DT) threshold_floor, threshold_ceil;
        constexpr uint32_t threshold_b_pos = NUM_THRESHOLDS / 2;
        // Set threshold_floor
        if (threshold_idx < threshold_b_pos) {
            // index 0 to threshold_b_pos - 1
            // [threshold_a, threshold_b]
            threshold_floor = threshold_a +
                              TO_DTYPE(DT, TO_FLOAT(DT, threshold_b - threshold_a) * threshold_idx / (threshold_b_pos - 1));
        } else {
            // index threshold_b_pos to NUM_THRESHOLDS - 1
            // (threshold_b, 0]
            threshold_floor = TO_DTYPE(DT, TO_FLOAT(DT, threshold_b) * (NUM_THRESHOLDS - threshold_idx - 1) /
                                          (NUM_THRESHOLDS - threshold_b_pos));
        }

        // Set threshold_ceil using shfl instructions
        threshold_ceil = __shfl_up_sync(0xFFFFFFFF, threshold_floor, values_per_threshold_per_warp);

        if (threshold_idx == 0) {
            threshold_ceil = TO_DTYPE(DT, INFINITY);
        }

        // -------------------------- Compare All Values Against All Thresholds --------------------------
        for (uint32_t row_offset = row_offset_start; row_offset < clipped_chunk_size; row_offset += idx_increment) {
            const uint32_t shifted_row_offset = (row_offset + warp_lane) % clipped_chunk_size;
            const uint32_t row_idx = chunk_start_idx + shifted_row_offset;
            FP_DTYPE(DT) activation = shared_activations[shifted_row_offset];
            // Compare the activation with the current thread's threshold
            const FP_DTYPE(DT) abs_activation = ABS(DT, activation);
            if (abs_activation >= threshold_floor && abs_activation < threshold_ceil) {
                // Atomic increment the count for this threshold
                const uint32_t idx = atomicAdd(&shared_counts[threshold_idx], 1);
                if (idx < adjusted_k_chunk) {
                    selected_rows_by_threshold[idx][threshold_idx] = row_idx;
                } else {
                    break;
                }
            }
        }

        __syncthreads();

        // -------------------------- Select Best Threshold Based on Counts --------------------------

        if (thread_idx < WARP_SIZE) {
            uint32_t count = 0;
            for (uint32_t threshold_idx = 0; threshold_idx < NUM_THRESHOLDS; threshold_idx++) {
                uint32_t threshold_count = shared_counts[threshold_idx];
                uint32_t remaining_rows = adjusted_k_chunk - count;
                uint32_t num_values_to_fetch = min(remaining_rows, threshold_count);
                for (uint32_t i = threadIdx.x; i < num_values_to_fetch; i += WARP_SIZE) {
                    const uint32_t selected_row = selected_rows_by_threshold[i][threshold_idx];
                    selected_rows[chunk_idx * k_chunk + count + i] = selected_row;
                    selected_activations[chunk_idx * k_chunk + count + i] = shared_activations[
                            selected_row - chunk_start_idx];
                }
                count += num_values_to_fetch;
                if (count == adjusted_k_chunk) {
                    break;
                }
            }
            assert(count ==
                   adjusted_k_chunk); // We should have exactly k_chunk rows
        }
        __syncthreads(); // ABSOLUTELY NECESSARY because we are reusing shared memory
    }

    grid.sync();

    ///////////////////////////////// Matrix Fetch & Dequant /////////////////////////////////
    // Throughput: 20 GB/s on PCIe 4.0 x16
    // Load selected rows and activations into shared memory

    const uint32_t num_matrix_cols = num_cols / NIBBLES_PER_ELEMENT;
    if (blockIdx.x * channels_per_tb >= num_matrix_cols) {
        // We are not using this threadblock for fetch
        return;
    }

    if (!reduced_shared_memory) {
        for (uint32_t selected_row_idx = threadIdx.y * blockDim.x + threadIdx.x;
             selected_row_idx < num_rows_to_fetch; selected_row_idx += blockDim.y * blockDim.x) {
            shared_selected_activations[selected_row_idx] = selected_activations[selected_row_idx];
            shared_selected_rows[selected_row_idx] = selected_rows[selected_row_idx];
        }
    }

    const uint32_t repeats = channels_per_tb / blockDim.x;

    for (uint32_t repeat_idx = 0; repeat_idx < repeats; repeat_idx++) {
        // Initialize unpacked to 0
        for (uint32_t nibble_idx = 0; nibble_idx < NIBBLES_PER_ELEMENT; nibble_idx++) {
            unpacked[threadIdx.y * blockDim.x * NIBBLES_PER_ELEMENT +
                     nibble_idx * blockDim.x +
                     threadIdx.x] = TO_DTYPE(DT, 0.0);
        }

        const uint32_t matrix_col_idx_start = blockIdx.x * channels_per_tb + repeat_idx * blockDim.x;

        const uint32_t matrix_col_offset = threadIdx.x;

        const uint32_t matrix_col_idx = matrix_col_idx_start + matrix_col_offset;
        const uint32_t thread_col_idx_start = matrix_col_idx * NIBBLES_PER_ELEMENT;

        // Load scales into shared memory
        for (uint32_t nibble_idx = threadIdx.y; nibble_idx < NIBBLES_PER_ELEMENT; nibble_idx += blockDim.y) {
            shared_reordered_scales[nibble_idx * blockDim.x + matrix_col_offset] = reordered_scales[
                    nibble_idx * num_matrix_cols + matrix_col_idx];
        }

        __syncthreads();

        if (matrix_col_idx < num_matrix_cols) {
            // Fetch & Dequantize
            for (uint32_t selected_row_idx = threadIdx.y;
                 selected_row_idx < num_rows_to_fetch; selected_row_idx += blockDim.y) {
                uint32_t row_to_fetch;
                FP_DTYPE(DT) activation{};
                if (!reduced_shared_memory) {
                    row_to_fetch = shared_selected_rows[selected_row_idx];
                    activation = shared_selected_activations[selected_row_idx];
                } else {
                    row_to_fetch = selected_rows[selected_row_idx];
                    activation = selected_activations[selected_row_idx];
                }
                // Zero-copy fetch
                const uint32_t fetched = cheatsheet[row_to_fetch * num_matrix_cols + matrix_col_idx];

                for (uint32_t nibble_idx = 0; nibble_idx < NIBBLES_PER_ELEMENT; nibble_idx++) {
                    const uint32_t col_idx = thread_col_idx_start + nibble_idx;
                    if (col_idx < num_cols) {
                        auto nibble = static_cast<FP_DTYPE(DT)>(
                                static_cast<uint8_t>((fetched >> (nibble_idx * BIT_WIDTH)) & ((1 << BIT_WIDTH) - 1)) -
                                (1 << (BIT_WIDTH - 1)));
                        const FP_DTYPE(DT) scale = shared_reordered_scales[nibble_idx * blockDim.x + matrix_col_offset];
                        unpacked[threadIdx.y * blockDim.x * NIBBLES_PER_ELEMENT + nibble_idx * blockDim.x +
                                 matrix_col_offset] += nibble * scale * activation;
                    }
                }
            }
        }

        __syncthreads();

        ///////////////////////////////// Reduction /////////////////////////////////
        // Latency: 1 us

        for (uint32_t col_offset = threadIdx.y * blockDim.x + threadIdx.x;
             col_offset < blockDim.x * NIBBLES_PER_ELEMENT; col_offset += blockDim.y * blockDim.x) {
            const uint32_t col_idx = matrix_col_idx_start * NIBBLES_PER_ELEMENT + col_offset;
            if (col_idx < num_cols) {
                FP_DTYPE(DT) col_sum = TO_DTYPE(DT, 0.0);
                for (uint32_t i = 0; i < blockDim.y; i++) {
                    //col_sum += unpacked[i][col_offset % NIBBLES_PER_ELEMENT][col_offset / NIBBLES_PER_ELEMENT];
                    col_sum += unpacked[i * blockDim.x * NIBBLES_PER_ELEMENT +
                                        col_offset % NIBBLES_PER_ELEMENT * blockDim.x +
                                        col_offset / NIBBLES_PER_ELEMENT];
                }
                ATOMIC_ADD(DT, &final_sum[col_idx], col_sum);
            }
        }
        __syncthreads();  // ABSOLUTELY NECESSARY because we are reusing shared memory
    }
}


// without the row selection part, for testing purposes
template<DataType DT>
__global__ void dec_kernel(
        const uint32_t *__restrict__ cheatsheet, const FP_DTYPE(DT) *__restrict__ activations,
        uint32_t *__restrict__ selected_rows,
        FP_DTYPE(DT) *__restrict__ final_sum, const FP_DTYPE(DT) *__restrict__ reordered_scales,
        const uint32_t num_rows, const uint32_t num_cols,
        const uint32_t num_chunks,
        const uint32_t chunk_size, const uint32_t k_chunk,
        const uint32_t channels_per_tb,
        const bool reduced_shared_memory
) {
    if (k_chunk == 0) {
        return;
    }

    ASSERT(blockDim.x <= channels_per_tb);
    ASSERT(channels_per_tb % blockDim.x == 0);
    ASSERT(blockDim.x % WARP_SIZE == 0); // For the row-selection to work

    // Handle num_rows_to_fetch, including the case where num_rows is not a multiple of chunk_size
    uint32_t num_rows_to_fetch = k_chunk * num_rows / chunk_size;

    // Kernel Launch Overhead: 2 us
    ///////////////////////////////// Shared Memory Initialization /////////////////////////////////
    __shared__ uint32_t shared_counts[NUM_THRESHOLDS];
    extern __shared__ unsigned char shared_memory[];
    auto (*selected_rows_by_threshold)[NUM_THRESHOLDS] = reinterpret_cast<uint32_t (*)[NUM_THRESHOLDS]>(shared_memory);
    auto *shared_activations = reinterpret_cast<FP_DTYPE(DT) *>(&selected_rows_by_threshold[k_chunk]);

    // Shared memory can be safely reused for the fetch phase
    uint32_t *shared_selected_rows;
    FP_DTYPE(DT) *shared_selected_activations, *unpacked, *shared_reordered_scales;
    if (!reduced_shared_memory) {
        // this must come first, for alignment
        shared_selected_rows = reinterpret_cast<uint32_t *>(&shared_memory);
        unpacked = reinterpret_cast<FP_DTYPE(DT) *>(&shared_selected_rows[num_rows_to_fetch]);
        shared_reordered_scales = reinterpret_cast<FP_DTYPE(DT) *>(&unpacked[blockDim.y * blockDim.x *
                                                                         NIBBLES_PER_ELEMENT]);
        shared_selected_activations = reinterpret_cast<FP_DTYPE(DT) *>(&shared_reordered_scales[blockDim.x *
                                                                                            NIBBLES_PER_ELEMENT]);

    } else {
        // don't use shared memory for selected rows and activations
        unpacked = reinterpret_cast<FP_DTYPE(DT) *>(&shared_memory);
        shared_reordered_scales = reinterpret_cast<FP_DTYPE(DT) *>(&unpacked[blockDim.y * blockDim.x *
                                                                         NIBBLES_PER_ELEMENT]);
    }

    ///////////////////////////////// Matrix Fetch & Dequant /////////////////////////////////
    // Throughput: 20 GB/s on PCIe 4.0 x16
    // Load selected rows and activations into shared memory

    const uint32_t num_matrix_cols = num_cols / NIBBLES_PER_ELEMENT;
    if (blockIdx.x * channels_per_tb >= num_matrix_cols) {
        // We are not using this threadblock for fetch
        return;
    }

    if (!reduced_shared_memory) {
        for (uint32_t selected_row_idx = threadIdx.y * blockDim.x + threadIdx.x;
             selected_row_idx < num_rows_to_fetch; selected_row_idx += blockDim.y * blockDim.x) {
            shared_selected_rows[selected_row_idx] = selected_rows[selected_row_idx];
            shared_selected_activations[selected_row_idx] = activations[selected_rows[selected_row_idx]];
        }
    }

    const uint32_t repeats = channels_per_tb / blockDim.x;

    for (uint32_t repeat_idx = 0; repeat_idx < repeats; repeat_idx++) {
        // Initialize unpacked to 0
        for (uint32_t nibble_idx = 0; nibble_idx < NIBBLES_PER_ELEMENT; nibble_idx++) {
            unpacked[threadIdx.y * blockDim.x * NIBBLES_PER_ELEMENT +
                     nibble_idx * blockDim.x +
                     threadIdx.x] = TO_DTYPE(DT, 0.0);
        }

        const uint32_t matrix_col_idx_start = blockIdx.x * channels_per_tb + repeat_idx * blockDim.x;

        const uint32_t matrix_col_offset = threadIdx.x;

        const uint32_t matrix_col_idx = matrix_col_idx_start + matrix_col_offset;
        const uint32_t thread_col_idx_start = matrix_col_idx * NIBBLES_PER_ELEMENT;

        // Load scales into shared memory
        for (uint32_t nibble_idx = threadIdx.y; nibble_idx < NIBBLES_PER_ELEMENT; nibble_idx += blockDim.y) {
            shared_reordered_scales[nibble_idx * blockDim.x + matrix_col_offset] = reordered_scales[
                    nibble_idx * num_matrix_cols + matrix_col_idx];
        }

        __syncthreads();

        if (matrix_col_idx < num_matrix_cols) {
            // Fetch & Dequantize
            for (uint32_t selected_row_idx = threadIdx.y;
                 selected_row_idx < num_rows_to_fetch; selected_row_idx += blockDim.y) {
                uint32_t row_to_fetch;
                FP_DTYPE(DT) activation{};
                if (!reduced_shared_memory) {
                    row_to_fetch = shared_selected_rows[selected_row_idx];
                    activation = shared_selected_activations[selected_row_idx];
                } else {
                    row_to_fetch = selected_rows[selected_row_idx];
                    activation = activations[row_to_fetch];
                }
                // Zero-copy fetch
                const uint32_t fetched = cheatsheet[row_to_fetch * num_matrix_cols + matrix_col_idx];

                for (uint32_t nibble_idx = 0; nibble_idx < NIBBLES_PER_ELEMENT; nibble_idx++) {
                    const uint32_t col_idx = thread_col_idx_start + nibble_idx;
                    if (col_idx < num_cols) {
                        auto nibble = static_cast<FP_DTYPE(DT)>(
                                static_cast<uint8_t>((fetched >> (nibble_idx * BIT_WIDTH)) & ((1 << BIT_WIDTH) - 1)) -
                                (1 << (BIT_WIDTH - 1)));
                        const FP_DTYPE(DT) scale = shared_reordered_scales[nibble_idx * blockDim.x + matrix_col_offset];
                        unpacked[threadIdx.y * blockDim.x * NIBBLES_PER_ELEMENT + nibble_idx * blockDim.x +
                                 matrix_col_offset] += nibble * scale * activation;
                    }
                }
            }
        }

        __syncthreads();

        ///////////////////////////////// Reduction /////////////////////////////////
        // Latency: 1 us

        for (uint32_t col_offset = threadIdx.y * blockDim.x + threadIdx.x;
             col_offset < blockDim.x * NIBBLES_PER_ELEMENT; col_offset += blockDim.y * blockDim.x) {
            const uint32_t col_idx = matrix_col_idx_start * NIBBLES_PER_ELEMENT + col_offset;
            if (col_idx < num_cols) {
                FP_DTYPE(DT) col_sum = TO_DTYPE(DT, 0.0);
                for (uint32_t i = 0; i < blockDim.y; i++) {
                    //col_sum += unpacked[i][col_offset % NIBBLES_PER_ELEMENT][col_offset / NIBBLES_PER_ELEMENT];
                    col_sum += unpacked[i * blockDim.x * NIBBLES_PER_ELEMENT +
                                        col_offset % NIBBLES_PER_ELEMENT * blockDim.x +
                                        col_offset / NIBBLES_PER_ELEMENT];
                }
                ATOMIC_ADD(DT, &final_sum[col_idx], col_sum);
            }
        }
        __syncthreads();  // ABSOLUTELY NECESSARY because we are reusing shared memory
    }
}
// Explicit template instantiation
#define INSTANTIATE_FOR_DATATYPE(DT) \
    template __global__ void fused_dec_kernel<DT>( \
        const uint32_t *__restrict__ cheatsheet, const FP_DTYPE(DT) *__restrict__ activations, \
        uint32_t *__restrict__ selected_rows, FP_DTYPE(DT) *__restrict__ selected_activations, \
        FP_DTYPE(DT) *__restrict__ final_sum, const FP_DTYPE(DT) *__restrict__ reordered_scales, \
        const FP_DTYPE(DT) threshold_a, const FP_DTYPE(DT) threshold_b, const uint32_t num_rows, const uint32_t num_cols, \
        const uint32_t num_chunks, \
        const uint32_t chunk_size, const uint32_t k_chunk, \
        const uint32_t channels_per_tb, \
        const bool reduced_shared_memory \
    ); \
    template __global__ void dec_kernel<DT>( \
        const uint32_t *__restrict__ cheatsheet, const FP_DTYPE(DT) *__restrict__ activations, \
        uint32_t *__restrict__ selected_rows, \
        FP_DTYPE(DT) *__restrict__ final_sum, const FP_DTYPE(DT) *__restrict__ reordered_scales, \
        const uint32_t num_rows, const uint32_t num_cols, \
        const uint32_t num_chunks, \
        const uint32_t chunk_size, const uint32_t k_chunk, \
        const uint32_t channels_per_tb, \
        const bool reduced_shared_memory \
    );

INSTANTIATE_FOR_DATATYPE(DataType::FP32)
INSTANTIATE_FOR_DATATYPE(DataType::FP16)
INSTANTIATE_FOR_DATATYPE(DataType::BF16)
