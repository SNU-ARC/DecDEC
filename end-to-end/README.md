# End-to-End LLM Inference with **DecDEC**

## Prerequisites
- A real-quantized model checkpoint (e.g., from SqueezeLLM or AWQ) (see [decdec_setup/](../decdec_setup/)).
- cheatsheet.pt and thresholds.pt (produced from cheatsheet.py and thresholds.py in [decdec_setup/](../decdec_setup/))
- DecDEC hyperparameters (k_chunk) from the [autotuner](../autotuner/)

## Quick Start

```bash
# Benchmark generation speed (5 runs, 1 000 output tokens each)
python generate.py --compile 2 --num_samples 5 \
       --model_name Meta-Llama-3-8B-Instruct --bitwidth 3 --dtype float16 \
       --checkpoint_path ./anyprec-(Meta-Llama-3-8B-Instruct)-w3_orig3-gc1-c4_s100_blk512 \
       --backend ap --n_tbs 12 --k_chunks 8 8 8 8 \
       --max_new_tokens 1000

# Generate 100 tokens once, printing the result
python generate.py --prompt "What is the capital of France?" --compile 2 --num_samples 1 \
       --model_name Meta-Llama-3-8B-Instruct --bitwidth 3 --dtype float16 \
       --checkpoint_path ./anyprec-(Meta-Llama-3-8B-Instruct)-w3_orig3-gc1-c4_s100_blk512 \
       --backend ap --n_tbs 12 --k_chunks 8 8 8 8 \
       --max_new_tokens 100 --print_result
```

## Command-Line Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt` | *str* | `None` | Text prompt. If omitted, a single BOS token is used. |
| `--num_samples` | int | `5` | Number of independent generations to run. |
| `--max_new_tokens` | int | `200` | Maximum number of tokens to generate **per sample** (not counting the prompt). |
| `--top_k` | int | `200` | Top-*k* nucleus filter applied before sampling (set to a large value to disable). |
| `--temperature` | float | `0.0` | Softmax temperature. `0` ≈ greedy decoding; > 0 enables stochastic sampling. |
| `--compile` | int | `2` | Torch-Inductor compilation mode: <br>• `0` — disable <br>• `1` — `max-autotune-no-cudagraphs` (safer) <br>• `2` — `max-autotune` + CUDA graphs after the first iteration. |
| `--device` | str | *auto* (`cuda` if available) | Target device ID (`cuda`, `cuda:1` …). |
| `--model_name` | str | **required** | Supported choices: `Meta-Llama-3-8B-Instruct`, `Phi-3-medium-4k-instruct`. |
| `--bitwidth` | `int` or `float` | **required** | Quantization bit-width per weight. Can be fractional (`3.5`) for mixed precision. |
| `--checkpoint_path` | str | **required** | Directory containing the quantized model weights, tokenizer, `cheatsheet.pt`, `thresholds.pt`, and (optionally) `bitwidth_map.json` for mixed precision. |
| `--dtype` | str | `float16` | GEMV compute dtype: `float16`, `bfloat16`, or `float32`. |
| `--backend` | str | **required** | GEMV kernel: `ap` (AnyPrec / APLinear) or `lutgemm` (LUT-GEMM). |
| `--n_tbs` | list[int] | **required** | $n_{tb}$, the number of thread blocks allocated for DEC. Provide **two** numbers for mixed-precision (one per bit-width). |
| `--k_chunks` | list[int] | **required** | $k_{chunk}$ hyper-parameters (one per QKV/O/GU/D projection). For mixed precision supply **8** integers—four for each bit-width. |
| `--print_result` | flag | _off_ | Print the decoded text for each sample. |

> ℹ️ **Mixed precision:** Supply fractional `--bitwidth` (e.g. `3.5`) **plus** matching `bitwidth_map.json`, two `n_tbs`, and eight `k_chunks`.

## Understanding `--compile`

| Value | Torch-Inductor Mode | CUDA Graphs | Recommended Use |
|-------|--------------------|-------------|-----------------|
| `0` | **off** | ✗ | Quick debugging. |
| `1` | `max-autotune-no-cudagraphs` | ✗ | First run after code changes. |
| `2` | `max-autotune` | ✓ (after warm-up) | Benchmarking & production. |

The first sample always runs without graphs to let Torch-Compile build the kernels; subsequent iterations replay a captured CUDA graph for maximal throughput.

## Output Metrics

After every sample, `generate.py` prints:

```
Time for inference 1: 5.06 sec total, 197.81 tokens/sec
```

and, at the end, the average tokens-per-second across all samples.
