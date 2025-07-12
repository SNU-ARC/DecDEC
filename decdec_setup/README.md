# DecDEC Setup Scripts

Setup scripts for deploying **DecDEC** on *weight-only, post-training quantized (PTQ)* language models that keep a one-to-one mapping between the quantized and original weights.

These scripts produce two binary blobs that DecDEC needs at inference:

| File | Purpose |
| ---- | ------- |
| **`cheatsheet.pt`** | Quantized & packed DecDEC residuals |
| **`thresholds.pt`** | Bucket boundaries for DecDEC’s approximate top-k selection |

## 0. Before You Start

### Key Terms

| Term | Definition |
|------|------------|
| **original model** | The full-precision model (FP16) before quantization and the baseline DecDEC tries to recover at inference time. |
| **fake-quantized model** | A model whose quantized weights have been **cast back to FP16**. It is only used while *building* DecDEC artifacts. |
| **real-quantized model** | The final integer / low-bit model used at inference. DecDEC consumes this plus the two auxiliary `.pt` files generated below. |

> **AWQ note**: Always treat the *scaled* full-precision model produced by AWQ as your “original model”. The un-scaled weights will not match the quantized ones.

## 1. Supported Quantization Methods

Any weight-only PTQ method that preserves the weight layout will work. We have verified:

- **SqueezeLLM**
- **AWQ**

## 2. Quantize Your Model First

DecDEC assumes that you *already* have:

1. The **original** (full-precision) model.
2. A **fake-quantized** copy of the same model.

Follow one of the guides below to obtain them.

> Note that this example uses the `meta-llama/Meta-Llama-3-8B-Instruct` model, but you can replace it with any other model that you wish to quantize. Please adjust the paths accordingly if you use a different model.

### 2.1 AWQ Example

```bash
cd llm-awq

./llama3_example.sh

./qwen2.5_example.sh

./phi_example.sh

# Running the scripts will produce the following files:
#   ↳ scaled_cache/$MODEL_NAME   – scaled full-precision model
#   ↳ fakequant_cache/$MODEL_NAME   – fake-quantized model
#   ↳ quant_cache/$MODEL_NAME   – quantized model
```

### 2.2 SqueezeLLM Example

We recommend using the Any-Precision LLM repository (included as a submodule) to quantize models in SqueezeLLM format. Although it supports multi-precision upscaling, only the seed model is needed for DecDEC, so `parent_precision` should match `seed_precision`. For inference, we also suggest using Any-Precision LLM's kernels, as they make better use of SM compute resources.

```bash
# Clone the repository
git submodule update --init --recursive

# Quantize to 3-bit (seed == parent precision → no upscaling)
cd any-precision-llm
python quantize.py meta-llama/Meta-Llama-3-8B-Instruct \
    --random_state 0 \
    --seed_precision 3 \
    --parent_precision 3
    #--cpu_only  # Uncomment to run gradient computation on CPU, if VRAM is limited

# The quantization will produce the following files:
#   ↳ cache/quantized   – intermediate weights
#   ↳ cache/packed      – final packed model

cd ..
# Convert to fake-quantized FP16
python conv_sqllm.py \
    --original_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --quantized_path "./any-precision-llm/cache/quantized/(Meta-Llama-3-8B-Instruct)-w3-orig3-gc1-c4_s100_blk512" \
    --output_path "./fakesqllm-(Meta-Llama-3-8B-Instruct)-w3_orig3-gc1-c4_s100_blk512"
```

Resulting paths:

| Model Type | Path / Identifier |
|------------|------------------|
| Original full-precision | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Fake-quantized | `./fakesqllm-(Meta-Llama-3-8B-Instruct)-w3_orig3-gc1-c4_s100_blk512` |
| Real-quantized | `./any-precision-llm/cache/packed/anyprec-(Meta-Llama-3-8B-Instruct)-w3-orig3-gc1-c4_s100_blk512` |

## 3. Prepare DecDEC Files

Once you have the original and fake-quantized models, you can run the setup scripts to generate the necessary DecDEC artifacts.

1. **`cheatsheet.pt`** – DecDEC residuals (4-bit packed by default).
2. **`thresholds.pt`** – Threshold vectors for approximate top-k.

High-level workflow:

| Step | Action | Script |
|------|--------|--------|
| 1 | Compare original vs. fake-quantized weights, grid-search scale, pack residuals | `cheatsheet.py` |
| 2 | Derive bucket boundaries for top-k approximation | `thresholds.py` |

## 4. Scripts Reference

| Script | Required Inputs | Main Output |
|--------|-----------------|-------------|
| **`cheatsheet.py`** | `--original_model` (FP16)<br>`--quantized_model` (fake-quantized FP16) | `cheatsheet.pt` |
| **`thresholds.py`** | `--model_path` (FP16) | `thresholds.pt` |

## 5. Run the Scripts

These examples assume having used SqueezeLLM to quantize the `meta-llama/Meta-Llama-3-8B-Instruct` model, resulting in the paths mentioned above. For AWQ, adjust the paths accordingly.

### 5.1 Create `cheatsheet.pt`

```bash
python cheatsheet.py \
    --original_model meta-llama/Meta-Llama-3-8B-Instruct \
    --quantized_model ./fakesqllm-(Meta-Llama-3-8B-Instruct)-w3_orig3-gc1-c4_s100_blk512 \
    --save_path .
```

Default is 4-bit packed residuals. See `--help` for alternative bit-widths or saving raw FP16 residuals.

### 5.2 Compute `thresholds.pt`

This scripts computes the top-k selection thresholds for DecDEC. Runs inference on GPU by default.

```bash
python thresholds.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --save_path .
    #--cpu_only  # Uncomment to run on CPU if VRAM is limited
```

## 6. What’s Next?

Place the generated `cheatsheet.pt` and `thresholds.pt` files in the same directory as your real-quantized model. DecDEC will automatically load them at inference time.

- **Autotune DecDEC Parameters** (`n_tb`, `k_chunk`) → [`autotuner/`](../autotuner/)
- **Accuracy Evaluation** → [`evaluation/`](../evaluation/)
- **Throughput Benchmark** → [`end-to-end/`](../end-to-end/)
