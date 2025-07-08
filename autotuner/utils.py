from numba import cuda
import subprocess
import sys
import re


def get_device_info():
    device = cuda.get_current_device()
    gpu_name = device.name.decode('utf-8')
    total_sm_count = device.MULTIPROCESSOR_COUNT
    max_available_shared_memory = device.MAX_SHARED_MEMORY_PER_BLOCK
    l2_cache_size = device.L2_CACHE_SIZE

    memory_clock_rate_khz = device.MEMORY_CLOCK_RATE
    memory_bus_width = device.GLOBAL_MEMORY_BUS_WIDTH 

    memory_bandwidth = memory_clock_rate_khz * memory_bus_width * 2 / 8 / 1e6

    return {
        'gpu_name': gpu_name,
        'total_sm_count': total_sm_count,
        'max_available_shared_memory': max_available_shared_memory,
        'l2_cache_size': l2_cache_size,
        'memory_bandwidth': memory_bandwidth,
    }


def get_pcie_info():
    # Run `nvidia-smi -q` and capture the output
    result = subprocess.run(['nvidia-smi', '-q'], stdout=subprocess.PIPE, text=True)
    output = result.stdout.splitlines()  # Split output into lines for easier processing

    pcie_gen_max = None
    pcie_lanes_max = None

    # Iterate through each line to find PCIe Generation and Link Width info
    for i, line in enumerate(output):
        if 'PCIe Generation' in line:
            # Look for the 'Max' line nearby for PCIe Generation
            for j in range(i + 1, len(output)):
                if 'Max' in output[j]:
                    pcie_gen_max = int(output[j].split(':')[-1].strip())
                    break

        elif 'Link Width' in line:
            # Look for the 'Max' line nearby for Link Width
            for j in range(i + 1, len(output)):
                if 'Max' in output[j]:
                    pcie_lanes_max = int(output[j].split(':')[-1].strip()[:-1])  # Remove the 'x' postfix
                    break

    return pcie_gen_max, pcie_lanes_max


def get_gemv_kernel_name(algo, bit_width):
    match algo:
        case 'sqllm':
            return f'matmul_kbit_32'
        case 'awq':
            return f'nqmv_bias'
        case _:
            raise ValueError(f"Unknown quantization algorithm: {algo}")


def get_output_path(prefix, model_name, fp_dtype, base_gemv, bit_width):
    return f"./nsys/{prefix}_{model_name}_{fp_dtype}_{base_gemv}_{bit_width}"

DEC_KERNEL_NAME = 'fused_dec_kernel'

