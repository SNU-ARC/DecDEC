# DecDEC: A Systems Approach to Advancing Low‑Bit LLM Quantization [[Paper](https://arxiv.org/pdf/2412.20185)]

![decdec](https://github.com/user-attachments/assets/4113a94c-46e4-45a7-ba3f-1fbd36a5ceae)

**DecDEC** (Decoding with **D**ynamic **E**rror **C**ompensation) is an *inference-time* add‑on that **restores the accuracy of aggressively quantized Large Language Models (LLMs)** while keeping their tiny memory footprint and fast latency.  

At run‑time the GPU fetches a *small, activation‑aware slice* of full‑precision weight **residuals** from CPU memory and applies them on‑the‑fly, recovering the lost accuracy of quantization with negligible memory and latency overhead.


## How DecDEC Works

1. **Channel Selection** – a fast bucket‑based Top‑K picks `k_chunk` channels (per 1024) with highest |activation|.  
2. **Residual Fetch** – 4‑bit residual rows for those channels are fetched from pinned CPU RAM via PCIe (CUDA zero‑copy).  
3. **Residual GEMV** – residuals are multiplied with the sparsified activation vector in a fused GPU kernel that runs parallel to the base GEMV.  
4. **Merge** – the residual output is atomically added to the GEMV result, yielding the compensated output.


## Performance at a Glance

| Model (AWQ 3‑bit) | GPU | PPL | Slow‑down |
|---------------|-----|-------|-----------|
| Llama‑3‑8B‑Instruct | RTX‑4050M | **9.41** (vs 10.49) | **+1.7 %** |
| Phi‑3‑medium-4k‑instruct  | RTX‑4070S | **5.23** (vs 5.92) | **+2.1 %** |

See full results in our paper.

## Repository Structure

This repository is organized as follows:

| Directory | What’s inside |
|-----------|---------------|
| **`decdec_setup/`**     | Utilities to quantize an LLM (3‑/3.5‑/4‑bit) and generate DecDEC artifacts (`cheatsheet.pt`, `thresholds.pt`). |
| **`decdec_extension/`** | A custom CUDA extension containing the fused *dynamic error‑compensation* kernel plus fast GEMV baselines. |
| **`autotuner/`**        | A two‑phase tuner that picks `n_tb` (thread‑blocks) and `k_chunk` (channels/1024‑chunk) to hit a target slow‑down bound. |
| **`end-to-end/`**       | Scripts to launch real‑time inference (prefill + decode) with DecDEC on single‑GPU desktops / laptops. |
| **`evaluation/`**       | Reproducible evaluation harness (WikiText‑2 perplexity, BBH accuracy, MT‑Bench) and plotting utilities. |

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SNU-ARC/decdec.git
   ```

2. **Install Dependencies**:
   Create a Python virtual environment and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **[`decdec_extension/`](decdec_extension/): Install the DecDEC CUDA Extension**:
    - Install the custom CUDA extension required by DecDEC, which includes the DecDEC kernel, base GEMV kernels for quantized inference, and some utility functions.

4. **[`decdec_setup/`](decdec_setup/): Prepare DecDEC Artifacts**:
   - Quantize your LLM and generate the required DecDEC artifacts (`cheatsheet.pt` and `thresholds.pt`).

5. You can now test and deploy DecDEC by exploring the following directories:

    **[`autotuner/`](autotuner/): Autotuning DecDEC Hyperparameters**:
    - Find optimal DecDEC hyperparameters ($n_{tb}$, $k_{chunk}$).

    **[`end-to-end/`](end-to-end/): End-to-End Deployment with DecDEC**:
    - Deploy your quantized model with DecDEC for real-time inference.

    **[`evaluation/`](evaluation/): Evaluate Quantized Models with DecDEC**:
    - Evaluate the accuracy of your quantized models using DecDEC on various tasks and datasets.

For each component, refer to the respective `README.md` files for detailed instructions and examples.

## A Note on Performance

DecDEC’s inference throughput is sensitive to GPU load and PCIe usage. For best results, avoid running graphical interfaces or other GPU-intensive processes during end-to-end inference and autotuning.

## Citing DecDEC

If you use DecDEC in your research, please cite:

```bibtex
@inproceedings{park-osdi25,
  title = {DecDEC: A Systems Approach to Advancing Low-Bit LLM Quantization},
  author = {Park, Yeonhong and Hyun, Jake and Kim, Hojoon and Jae W., Lee},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year = {2025}
}

## License

(TODO: Add license information here)
