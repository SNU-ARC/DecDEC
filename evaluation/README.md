# DecDEC Evaluation Scripts

This directory contains scripts to evaluate **quantized LLMs with DecDEC** using either:
- **LM-Eval-Harness tasks** (e.g., `bbh_cot_fewshot_causal_judgement`)
- **Perplexity** on standard datasets (e.g., `wikitext2`, `ptb`, `c4`)

This is a **reference implementation** for testing and analysing model accuracy, and doesn't execute DEC concurrently with the base GEMV operations. For real end-to-end deployment, see [`../end-to-end/`](../end-to-end/).

## Usage

Once you have:
- A real-quantized model (e.g., from SqueezeLLM or AWQ)
- `cheatsheet.pt` and `thresholds.pt` (produced from `cheatsheet.py` and `thresholds.py` in [`decdec_setup/`](../decdec_setup/))
- DecDEC hyperparameters (`k_chunk`) from the [autotuner](../autotuner/)

You can refer to [`run_eval.py`](./run_eval.py) for an example on how to:
- Evaluate a full-precision FP16 model as a baseline
- Load a quantized model with DecDEC
- Set different `k_chunk` values and evaluation modes
- Run LM-eval tasks or perplexity evaluation

`run_eval.py` uses utility functions from `evaltools.py`.

## Prerequisites

Before running the evaluation, install the custom CUDA extension required by DecDEC by referring to the instructions in [`decdec_extension/README.md`](../decdec_extension/README.md).
