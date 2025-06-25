# DecDEC: A Systems Approach to Advancing Low-Bit LLM Quantization

## Overview

This repository contains the following 5 main components:

| Directory | Description |
|-----------|-------------|
| [`decdec_setup/`](decdec_setup/) | Scripts to prepare quantized models and DecDEC artifacts. |
| [`decdec_extension/`](decdec_extension/) | Custom CUDA extension implementing the DecDEC kernel. |
| [`autotuner/`](autotuner/) | Autotuning scripts to find optimal DecDEC hyperparameters. |
| [`end-to-end/`](end-to-end/) | End-to-end deployment scripts for real-time inference with DecDEC. |
| [`evaluation/`](evaluation/) | Scripts to evaluate quantized LLMs with DecDEC. |

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

## Citation
(To be added)
