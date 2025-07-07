import subprocess
import time
import os
import sys
import config
from itertools import product
import utils

gpu_info = utils.get_device_info()

############################################################
# max_n_tb and max_k_chunk determination

# Calculating max_k_chunk based on GPU memory bandwidth
pcie_gen, pcie_lanes = utils.get_pcie_info()
print(f"Detected PCIe {pcie_gen}.0 x{pcie_lanes}")

_pcie_banwidth = 1 / 8 * (2 ** pcie_gen) * pcie_lanes  # 32 GB/s for PCIe 4.0 x16
_bandwidth_ratio = gpu_info['memory_bandwidth'] / _pcie_banwidth

print(gpu_info['memory_bandwidth'])

_residual_bitwidth = 4
_chunk_size = 1024
_k_chunk_estimate = int(max(config.bitwidths) / _bandwidth_ratio / _residual_bitwidth * _chunk_size)

max_k_chunk = _k_chunk_estimate * 2  # multiply by 2 to be safe

# We will use half of the total SMs for DEC at most
min_n_tb = 4  # because starting at 1 is too slow sometimes
max_n_tb = gpu_info['total_sm_count'] // 2

############################################################

os.makedirs("./nsys", exist_ok=True)

# receive prefix from command line
if len(sys.argv) < 2:
    print("Usage: python run_profile.py <prefix>")
    sys.exit(1)

num_configs = len(config.model_configurations) * len(config.algos) * len(config.bitwidths)
progress = 0
prefix = sys.argv[1]

for model_name, algo, bitwidth in product(config.model_configurations.keys(), config.algos, config.bitwidths):
    progress += 1
    print("======================================================================")
    print(f"AUTOTUNER Progress: {progress}/{num_configs} ({round(progress / num_configs * 100)}%)")

    halve_layers = config.halve_layers(model_name, algo, bitwidth)

    tokens_per_config = config.tokens_per_config
    if halve_layers:
        print(">> Halving the number of layers to prevent OOM <<")
        print(">> Doubling the number of tokens per config to compensate <<")
        tokens_per_config = config.tokens_per_config * 2

    output_path = config.get_output_path(prefix, model_name, config.fp_dtype, algo, bitwidth)

    # check if nsys output file already exists
    if os.path.exists(f"{output_path}.trace"):
        print(f"Output file {output_path}.trace already exists, skipping...")
        continue

    checkpoint_path = config.model_configurations[model_name]['checkpoint_path']

    command = (f"{config.nsys_path} profile --cuda-graph-trace=node --output={output_path} --export=sqlite --force-overwrite=true -s none --cpuctxsw=none "  # temporarily disable all traces to isolate bug
               f"python3 {config.profiler_path} --dtype {config.fp_dtype} --quant_algo {algo} --bitwidth {bitwidth} --model_name {model_name} --checkpoint_path '{checkpoint_path}' --min_n_tb {min_n_tb} --max_n_tb {max_n_tb} --max_k_chunk {max_k_chunk} --tokens_per_config {tokens_per_config}  --trace_file {output_path + '.trace'} {'--halve_layers' if halve_layers else ''}")

    print("Command: ", command)
    print("======================================================================")
    # Run the command and display output
    process = subprocess.run(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)

    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")

print("======================================================================")
print("All done!")
