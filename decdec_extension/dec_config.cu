#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include "typetraits.h"
#include "datatype.h"

#include "macros.h"
#include "dec_config.h"

// Utility functions

uint32_t ceiling_divide(uint32_t num, uint32_t den) {
    return (num + den - 1) / den;
}

uint32_t round_up_to_multiple(uint32_t num, uint32_t multiple) {
    return ceiling_divide(num, multiple) * multiple;
}

int get_shared_memory_per_block() {                                                                      
    int device;                                                                                          
    cudaGetDevice(&device);                                                                              
    int sharedMemPerBlock;                                                                               
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);              
    return sharedMemPerBlock;                                                                            
}                                                                                                        

int setup_dec_parameters(
        uint32_t num_rows,
        uint32_t num_cols,
        uint32_t num_threadblocks,
        uint32_t *num_chunks,
        uint32_t *chunk_size,
        uint32_t *channels_per_tb,
        dim3 *grid_dim,
        dim3 *block_dim
) {
    uint32_t num_total_elements = num_cols / NIBBLES_PER_ELEMENT;
    *channels_per_tb = (num_total_elements + num_threadblocks - 1) / num_threadblocks;

    // Round channels_per_tb up to the nearest multiple of WARP_SIZE
    *channels_per_tb = round_up_to_multiple(*channels_per_tb, WARP_SIZE);

    uint32_t block_dim_x;
    if (*channels_per_tb > MAX_MATRIX_COLS_PER_BLOCK) {
        uint32_t how_many_times_too_large = ceiling_divide(*channels_per_tb, MAX_MATRIX_COLS_PER_BLOCK);
        block_dim_x = num_total_elements / num_threadblocks / how_many_times_too_large;
        block_dim_x = round_up_to_multiple(block_dim_x, WARP_SIZE);
        *channels_per_tb = block_dim_x * how_many_times_too_large;
    } else {
        block_dim_x = *channels_per_tb;
    }

    uint32_t matrix_cols = (num_cols / NIBBLES_PER_ELEMENT);
    uint32_t num_blocks = ceiling_divide(matrix_cols, *channels_per_tb);
    uint32_t block_dim_y = THREADS_PER_BLOCK / block_dim_x;

    // Set chunk size and number of chunks
    *num_chunks = ceiling_divide(num_rows, CHUNK_SIZE);
    *chunk_size = CHUNK_SIZE;

    uint32_t max_useful_threadblocks = std::max(*num_chunks, num_blocks);

    *grid_dim = dim3(std::min(max_useful_threadblocks, num_threadblocks));
    *block_dim = dim3(block_dim_x, block_dim_y);

    return 0;
}

uint32_t get_required_shared_memory_size(
        DataType data_type,
        uint32_t max_available_shared_memory, int *shared_memory_status,
        uint32_t k_chunk, uint32_t num_chunks, uint32_t chunk_size, dim3 block_dim
) {
    uint32_t num_rows_to_fetch = k_chunk * num_chunks;

    uint32_t row_select_shared_memory_size = k_chunk * NUM_THRESHOLDS * sizeof(uint32_t) +
                                             chunk_size * SIZEOF(data_type);

    uint32_t dec_shared_memory_requirement = num_rows_to_fetch * sizeof(uint32_t) +  // shared_selected_rows
                                             NIBBLES_PER_ELEMENT * block_dim.x * block_dim.y * SIZEOF(data_type) // unpacked
                                             + block_dim.x * NIBBLES_PER_ELEMENT *
                                               SIZEOF(data_type) // shared_reordered_scales
                                             + num_rows_to_fetch * SIZEOF(data_type);  // shared_selected_activations

    uint32_t static_shared_memory_size = sizeof(uint32_t) * NUM_THRESHOLDS; // shared_counts

    uint32_t dynamic_shared_memory_size = std::max(row_select_shared_memory_size, dec_shared_memory_requirement);

    if (dynamic_shared_memory_size + static_shared_memory_size > max_available_shared_memory) {
        // Try the reduced memory version
        dec_shared_memory_requirement = NIBBLES_PER_ELEMENT * block_dim.x * block_dim.y * SIZEOF(data_type)  // unpacked
                                        + block_dim.x * NIBBLES_PER_ELEMENT * SIZEOF(data_type); // shared_reordered_scales
        dynamic_shared_memory_size = std::max(row_select_shared_memory_size, dec_shared_memory_requirement);

        if (dynamic_shared_memory_size + static_shared_memory_size > max_available_shared_memory) {
            // Not enough shared memory
            *shared_memory_status = -1;
        } else {
            // Reduced shared memory
            *shared_memory_status = 1;
        }
    } else {
        // Enough shared memory
        *shared_memory_status = 0;
    }

    return dynamic_shared_memory_size;
}


// Constructor
DECConfig::DECConfig(
    DataType _data_type,
    DECContext* _dec_context,
    uint32_t* _q_residual,
    void* _reordered_scales,
    void* _thresholds,
    uint32_t _num_rows,
    uint32_t _num_cols,
    uint32_t _num_q_residual_cols
) :
    data_type(_data_type),
    dec_context(_dec_context),
    q_residual(_q_residual),
    reordered_scales(_reordered_scales),
    thresholds(_thresholds),
    num_rows(_num_rows),
    num_cols(_num_cols),
    num_q_residual_cols(_num_q_residual_cols)
{
    cudaEventCreate(&event_1);
    cudaEventCreate(&event_2);

    TORCH_CHECK(data_type == dec_context->data_type, "Data type mismatch between DECConfig and DECContext.");
}

// Destructor
DECConfig::~DECConfig() {
    cudaEventDestroy(event_1);
    cudaEventDestroy(event_2);

    // Free zero-copy memory
    if (q_residual != nullptr) {
        cudaFreeHost(q_residual);
    }
    if (reordered_scales != nullptr) {
        cudaFreeHost(reordered_scales);
    }
}

// Setup function
void DECConfig::setup(uint32_t _k_chunk) {
    k_chunk = _k_chunk;

    // Setup configurations
    int valid = setup_dec_parameters(
        num_rows, num_cols, dec_context->num_tb, &num_chunks, 
        &chunk_size, &channels_per_tb, &grid_dim, &block_dim
    );
    assert(valid == 0);

    int shared_memory_status;
    uint32_t max_available_shared_memory = get_shared_memory_per_block();
    shared_memory_size = get_required_shared_memory_size(data_type,
        max_available_shared_memory, &shared_memory_status,
        k_chunk, num_chunks, chunk_size,
        block_dim
    );
    assert(shared_memory_status != -1);
    reduced_shared_memory = (shared_memory_status == 1);

    // Check that the buffer sizes are large enough
    uint32_t k = k_chunk * num_rows / chunk_size;
    TORCH_CHECK(dec_context->buffer_size >= k, "DECContext buffer size is too small for the given k_chunk.");
}

// create_dec_config_templated

template <DataType DT>
uintptr_t create_dec_config_templated(
    uintptr_t dec_context_uintptr,
    uint32_t k_chunk,
    torch::Tensor q_residual,
    torch::Tensor reordered_scales,
    torch::Tensor thresholds,
    DataType data_type
) {
    auto* dec_context = reinterpret_cast<DECContext*>(dec_context_uintptr);
    uint32_t num_rows = q_residual.size(0);
    uint32_t num_cols = reordered_scales.size(0);
    uint32_t num_q_residual_cols = q_residual.size(1);
    assert(num_q_residual_cols == num_cols / NIBBLES_PER_ELEMENT);

    // Setup zero-copy data (q_residual / reordered_scales)
    uint32_t *h_q_residual = nullptr, *d_q_residual = nullptr;
    if (q_residual.is_cpu()) {
        cudaHostAlloc((void**)&h_q_residual, num_rows * num_q_residual_cols * sizeof(uint32_t), 
                      cudaHostAllocMapped);
        cudaHostGetDevicePointer((void**)&d_q_residual, h_q_residual, 0);
        memcpy(h_q_residual, (uint32_t*)q_residual.data_ptr<int32_t>(), 
               num_rows * num_q_residual_cols * sizeof(uint32_t));
    } else {
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("DecDEC WARNING: q_residual is on GPU. This is NOT the intended setup.\n");
        printf("Zero-copy memory will NOT be used, and q_residual will be kept on GPU.\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        d_q_residual = (uint32_t*)q_residual.data_ptr<int32_t>();
    }

    FP_DTYPE(DT) *h_reordered_scales = nullptr, *d_reordered_scales = nullptr;
    if (reordered_scales.is_cpu()) {
        cudaHostAlloc((void**)&h_reordered_scales, num_cols * sizeof(FP_DTYPE(DT)), cudaHostAllocMapped);
        cudaHostGetDevicePointer((void**)&d_reordered_scales, h_reordered_scales, 0);
        memcpy(h_reordered_scales, (FP_DTYPE(DT)*)reordered_scales.data_ptr<ATEN_DTYPE(DT)>(), num_cols * sizeof(FP_DTYPE(DT)));
    } else {
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("DecDEC WARNING: reordered_scales is on GPU. This is NOT the intended setup.\n");
        printf("Zero-copy memory will NOT be used, and reordered_scales will be kept on GPU.\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        d_reordered_scales = (FP_DTYPE(DT)*)reordered_scales.data_ptr<ATEN_DTYPE(DT)>();
    }

    // Create DECConfig
    auto* dec_config = new DECConfig(
        data_type,
        dec_context,
        (uint32_t*)d_q_residual,
        (void*)d_reordered_scales,
        (void*)thresholds.data_ptr<ATEN_DTYPE(DT)>(),
        num_rows, num_cols, num_q_residual_cols
    );

    // Setup dec_config
    dec_config->setup(k_chunk);

    return reinterpret_cast<uintptr_t>(dec_config);
}

// create_dec_config
uintptr_t create_dec_config(
    uintptr_t dec_context_uintptr,
    uint32_t k_chunk,
    torch::Tensor q_residual,
    torch::Tensor reordered_scales,
    torch::Tensor thresholds
) {
    // Check that thresholds are on CPU
    TORCH_CHECK(thresholds.is_cpu(), "thresholds must be on CPU.");

    // Check that reordered_scales and thresholds have the same data type
    TORCH_CHECK(reordered_scales.scalar_type() == thresholds.scalar_type(),
        "reordered_scales and thresholds must have the same data type.");

    // Check that q_residual is of type int
    TORCH_CHECK(q_residual.scalar_type() == at::kInt, "q_residual must be of type int.");

    // Check that all tensors are contiguous
    TORCH_CHECK(q_residual.is_contiguous(), "q_residual must be contiguous.");
    TORCH_CHECK(reordered_scales.is_contiguous(), "reordered_scales must be contiguous.");
    TORCH_CHECK(thresholds.is_contiguous(), "thresholds must be contiguous.");

    // Determine the data type from reordered_scales and dispatch
    auto type = reordered_scales.scalar_type();
    DataType data_type;

    if (type == at::kFloat) {
        data_type = DataType::FP32;
        return create_dec_config_templated<DataType::FP32>(
            dec_context_uintptr, k_chunk, q_residual, reordered_scales, thresholds, data_type
        );
    } else if (type == at::kHalf) {
        data_type = DataType::FP16;
        return create_dec_config_templated<DataType::FP16>(
            dec_context_uintptr, k_chunk, q_residual, reordered_scales, thresholds, data_type
        );
    } else if (type == at::kBFloat16) {
        data_type = DataType::BF16;
        return create_dec_config_templated<DataType::BF16>(
            dec_context_uintptr, k_chunk, q_residual, reordered_scales, thresholds, data_type
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type in create_dec_config.");
        return 0;
    }
}

// update_dec_config
void update_dec_config(
    uintptr_t dec_config_uintptr,
    uintptr_t new_dec_context_uintptr,
    uint32_t k_chunk
) {
    auto* dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);
    DataType data_type = dec_config->data_type;

    // Re-cast to the correct type and update the DECContext
    switch (data_type) {
        case DataType::FP32: {
            auto* new_dec_context = reinterpret_cast<DECContext*>(new_dec_context_uintptr);
            dec_config->dec_context = new_dec_context;
            dec_config->setup(k_chunk);
            break;
        }
        case DataType::FP16: {
            auto* new_dec_context = reinterpret_cast<DECContext*>(new_dec_context_uintptr);
            dec_config->dec_context = new_dec_context;
            dec_config->setup(k_chunk);
            break;
        }
        case DataType::BF16: {
            auto* new_dec_context = reinterpret_cast<DECContext*>(new_dec_context_uintptr);
            dec_config->dec_context = new_dec_context;
            dec_config->setup(k_chunk);
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported data type in update_dec_config.");
    }
}

// destroy_dec_config
void destroy_dec_config(
    uintptr_t dec_config_uintptr
) {
    auto* dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);
    delete dec_config;
}

// config
std::unordered_map<std::string, int> read_dec_config(
    uintptr_t dec_config_uintptr
) {
    auto* dec_config = reinterpret_cast<DECConfig*>(dec_config_uintptr);

    std::unordered_map<std::string, int> config;
    config["num_chunks"] = dec_config->num_chunks;
    config["chunk_size"] = dec_config->chunk_size;
    config["k_chunk"] = dec_config->k_chunk;
    config["channels_per_tb"] = dec_config->channels_per_tb;
    config["n_tb"] = dec_config->grid_dim.x;
    config["block_dim_x"] = dec_config->block_dim.x;
    config["block_dim_y"] = dec_config->block_dim.y;
    config["shared_memory_size"] = dec_config->shared_memory_size;
    config["reduced_shared_memory"] = dec_config->reduced_shared_memory;

    return config;
}

