# DecDEC Profiling & Tuning Toolkit

This repository contains two thin CLI wrappers that make it easy to

1. **Collect Nsight‑Systems traces** for a single quantized model (`run_profile.py`)
2. **Analyse & autotune** DecDEC kernel parameters for that trace (`tuner.py`)

At the end of the process, you will have an optimal $n_{tb}$ value and $k_{chunk}$ parameters for each projection matrix, which you can use to run DecDEC with optimal performance.

## Prerequisites

- A real-quantized model checkpoint (e.g., from SqueezeLLM or AWQ) (see [decdec_setup/](../decdec_setup/)).
- cheatsheet.pt and thresholds.pt (produced from cheatsheet.py and thresholds.py in [decdec_setup/](../decdec_setup/))

## 1.  Collect Nsight traces

Below is an example of how to collect a trace for the `Meta-Llama-3-8B-Instruct` model using the `sqllm` algorithm with 3-bit quantization. The trace will be saved in the `./nsys/` directory.

```bash
python run_profile.py YOUR_PREFIX \
  --model_name Meta-Llama-3-8B-Instruct --algo sqllm --bitwidth 3 \
  --checkpoint_path "../end-to-end/anyprec-(Meta-Llama-3-8B-Instruct)-w3_orig3-gc1-c4_s100_blk512" \
  --fp_dtype fp16 --tokens 10

```

Arguments of note:

| flag | meaning | typical value |
|------|---------|---------------|
| `--model_name` | Model to profile (see below) | `Meta-Llama-3-8B-Instruct` or `Phi-3-medium-4k-instruct` or `Meta-Llama-3-70B-Instruct` |
| `--algo` | DecDEC algorithm to profile | `sqllm` or `awq` |
| `--bitwidth` | Quantization bit-width | `3` or `4` |
| `--tokens` | Tokens *per config* during micro‑bench | `10` |
| `--overwrite` | Re‑profile even if `./nsys/<prefix>.trace` already exists |  flag |

In our paper, we used 10 tokens per configurations as default.

> **Note**: The prefix is only used to name the output files. Use it to distinguish between different runs.

Each call generates:

```
./nsys/<PREFIX>_<MODEL>_<FP>_<ALGO>_<BIT>.trace   (raw trace)
./nsys/<…>.sqlite                                 (parsed database)
```

---

## 2. Run the tuner

After a trace is collected, launch **`tuner.py`** with identical identifiers **plus** a target slowdown rate as a percentage, and the four projection matrix dimensions:

### Projection dimensions

| model | QKV | O | GU | D |
|-------|-----|---|----|---|
| **Meta‑Llama‑3‑8B‑Instruct** | `4096 6144` | `4096 4096` | `4096 28672` | `14336 4096` |
| **Phi‑3‑medium‑4k‑instruct** | `5120 7680` | `5120 5120` | `5120 35840` | `17920 5120` |
| **Meta‑Llama‑3‑70B‑Instruct** | `8192 10240` | `8192 8192` | `8192 57344` | `28672 8192` |

### Example

```bash
python tuner.py YOUR_PREFIX \
  --fp_dtype fp16 \
  --model_name Meta-Llama-3-8B-Instruct \
  --algo sqllm --bitwidth 3 \
  --qkv 4096 6144 --o 4096 4096 --gu 4096 28672 --d 14336 4096
  -- target_slowdown 2.5
```

The tuner:

1. Parses the trace (`parse_results`)
2. Saves per‑kernel timings as `./csv/<prefix>_<model>_…csv`
3. Plots `./plots/<prefix>_<model>_…png`
4. Prints optimal `n_tb`, and `k_chunk` parameters for each projection matrix


## 3. Interpreting tuner output

The following is an example output from the tuning process:
```
========================== Fine-grained Search ===========================
Fine-grained k_chunk:
    qkv (4096 x 6144): 4 / 1024 (n_tb=12)
    o (4096 x 4096): 4 / 1024 (n_tb=8)
    gu (4096 x 28672): 5 / 1024 (n_tb=14)
    d (14336 x 4096): 5 / 1024 (n_tb=14)
Fine-grained layer time: 104.53 us (Slowdown: 1.10)
    qkv (4096 x 6144): 16.01 us 
    o (4096 x 4096): 13.08 us 
    gu (4096 x 28672): 49.29 us 
    d (14336 x 4096): 26.15 us 
========================== End Tuning ===========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tuning Results: Meta-Llama-3-8B-Instruct sqllm 3-bit @ 10.0% slowdown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Target Slowdown          : 10.0 %
Best n_tb                : 14
k_chunk per module       : 4, 4, 5, 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

* $k_{chunk}$ is presented as `k_chunk / 1024` as 1024 is the chunk size in the current implementation.
* `n_tb` is the chosen **maximum thread‑blocks** allocated for DEC.
* The four $k_{chunk}$ values correspond to the four projection matrices: QKV, O, GU, and D.

## 4. Next Steps

With the optimal parameters in hand you can:

- Benchmark inference throughput in [`end-to-end/`](../end-to-end/)
- Evaluate accuracy in [`evaluation/`](../evaluation/)
