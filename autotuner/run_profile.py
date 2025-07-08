import argparse, os, subprocess, sys
import utils

# ────────────────────────── CLI ────────────────────────── #
p = argparse.ArgumentParser(description="Profile DecDEC with Nsight Systems")
p.add_argument("prefix", help="Prefix for ./nsys output files")
p.add_argument("--model_name", required=True)
p.add_argument("--algo", required=True, choices=["sqllm", "awq"])
p.add_argument("--bitwidth", required=True, type=int, choices=[3, 4])
p.add_argument("--checkpoint_path", required=True)

p.add_argument("--fp_dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
p.add_argument("--tokens", type=int, default=10)
p.add_argument("--halve_layers", action="store_true")
p.add_argument("--overwrite", action="store_true")
p.add_argument("--min_n_tb", type=int, default=4)
p.add_argument("--max_n_tb", type=int)
p.add_argument("--nsys_path", default='nsys')
args = p.parse_args()

# ───────────────────────── GPU / PCIe info ───────────────────────── #
gpu = utils.get_device_info()
pcie_gen, pcie_lanes = utils.get_pcie_info()
print(f"Detected PCIe {pcie_gen}.0 x{pcie_lanes}")

pcie_bw = (1 / 8) * (2 ** pcie_gen) * pcie_lanes
bw_ratio = gpu["memory_bandwidth"] / pcie_bw
chunk_size, residual_bw = 1024, 4
k_chunk_est = int(args.bitwidth / bw_ratio / residual_bw * chunk_size)
max_k_chunk = k_chunk_est * 2

min_n_tb = args.min_n_tb
max_n_tb = args.max_n_tb or gpu["total_sm_count"] // 2
tokens_per_cfg = args.tokens * (2 if args.halve_layers else 1)

# ─────────────────────── Output path / skipping ─────────────────────── #
os.makedirs("./nsys", exist_ok=True)
output_path = utils.get_output_path(args.prefix, args.model_name,
                                    args.fp_dtype, args.algo, args.bitwidth)
trace_file = f"{output_path}.trace"
if os.path.exists(trace_file) and not args.overwrite:
    print(f"{trace_file} exists; use --overwrite to regenerate.")
    sys.exit(0)

# ───────────────────────── Build command ───────────────────────── #
command = (
    f"{args.nsys_path} profile --cuda-graph-trace=node "
    f"--output={output_path} --export=sqlite --force-overwrite=true "
    f"-s none --cpuctxsw=none "
    f"python3 ../end-to-end/profiler.py "
    f"--dtype {args.fp_dtype} --quant_algo {args.algo} --bitwidth {args.bitwidth} "
    f"--model_name {args.model_name} --checkpoint_path '{args.checkpoint_path}' "
    f"--min_n_tb {min_n_tb} --max_n_tb {max_n_tb} --max_k_chunk {max_k_chunk} "
    f"--tokens_per_config {tokens_per_cfg} --trace_file {trace_file} "
    f"{'--halve_layers' if args.halve_layers else ''}"
)

print("=" * 70)
print("Launching:")
print(command)
print("=" * 70)

res = subprocess.run(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
if res.returncode != 0:
    print(f"Command failed with return code {res.returncode}")
else:
    print("Profiling completed successfully!")