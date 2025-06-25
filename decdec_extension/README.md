# DecDEC PyTorch Extension

## Overview

This directory contains the source code for the DecDEC PyTorch extension. It provides interfaces and utilities for deploying DecDEC, including:

1. **Launching base quantized GEMV kernels**
2. **Configuring DecDEC runtime parameters**
3. **Launching DecDEC dynamic error compensation kernels**

> **Note:** In this extension, "DEC" stands for the Dynamic Error Compensation kernel, which corrects quantization errors during computation. "DecDEC" refers to the combined process of running the DEC kernel together with a base GEMV (General Matrix-Vector multiplication) kernel for improved accuracy during inference.

## Installation

You need to have a CUDA-enabled PyTorch environment to use this extension. The extension is built using C++ and CUDA, and it uses PyBind11 for Python bindings.

Make sure you have NVCC and g++ installed on your system.

To install the DecDEC extension, run the following command from this directory:

```bash
pip install .
```

> **Note:** You may need to modify NVCC flags in `setup.py` to match the compute capabilities of your target GPU.

## Usage

To use the DecDEC extension in your PyTorch project, import it as follows:

```python
import decdec_ext
```

The extension exposes several functions through its C++/CUDA backend via PyBind11. These functions fall into the following categories:

### 1. DECConfig Management

```python
decdec_ext.create_dec_config(
    dec_context_ptr, k_chunk,
    q_residual, reordered_scales, thresholds
) -> int (dec_config_ptr)

decdec_ext.read_dec_config(dec_config_ptr) -> dict

decdec_ext.update_dec_config(
    dec_config_ptr, new_dec_context_ptr, k_chunk
)

decdec_ext.destroy_dec_config(dec_config_ptr)
```

These functions manage the `DECConfig` object, which holds runtime parameters for dynamic error compensation.

### 2. Standalone DEC Kernels

```python
decdec_ext.dec(dec_config_ptr, input, output)
decdec_ext.unfused_dec(dec_config_ptr, input, output)
```

- `dec`: Launches the full DEC kernel.
- `unfused_dec`: Runs DEC without row selection for testing or ablation studies.

### 3. Base GEMV Kernels

```python
decdec_ext.anyprec_gemv(
    input, output, qweight, lut, bitwidth
)

decdec_ext.lutgemm_gemv(
    input, output, q_weight, alpha, q_bias,
    bitwidth, group_size
)

decdec_ext.sqllm_gemv(
    input, output, qweight, lut, bitwidth
)
```

These functions implement quantized GEMV (matrix-vector multiplication) kernels:

- **ANYPREC:** A table-based GEMV kernel supporting arbitrary bitwidths.  
  Adapted from [any-precision-llm](https://github.com/SNU-ARC/any-precision-llm)

- **SQLLM:** A structured lookup-based GEMV kernel optimized for SqueezeLLM-style quantization.  
  Adapted from [SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM)

- **LUTGEMM:** A LUT-based GEMV kernel with linear scaling, compatible with group-wise quantization.  
  Adapted from [lut-gemm](https://github.com/naver-aics/lut-gemm)

**Compatibility Notes:**

- The **ANYPREC** and **SQLLM** kernels are compatible with SqueezeLLM-style quantization.
- The **LUTGEMM** kernel supports both LUT-GEMM and AWQ quantization formats.

**Kernel Selection in Our Paper:**

In our paper, we used:
- **ANYPREC** for evaluating SqueezeLLM quantization
- **LUTGEMM** for evaluating AWQ quantization

**Rationale:**
- For **SqueezeLLM quantization**, we selected **ANYPREC** over the original SQLLM kernel because it offers comparable performance while being more efficient in SM (streaming multiprocessor) usage.
- For **AWQ quantization**, we used **LUTGEMM** to support both 3-bit and 4-bit quantization. The original AWQ kernel supports only 4-bit.

### 4. DEC + GEMV Combined Kernels

```python
decdec_ext.dec_anyprec(
    dec_config_ptr, input, output,
    qweight, lut, bitwidth
)

decdec_ext.dec_lutgemm(
    dec_config_ptr, input, output,
    q_weight, alpha, q_bias,
    bitwidth, group_size
)

decdec_ext.dec_sqllm(
    dec_config_ptr, input, output,
    q_weight, lut, bitwidth
)
```

These interfaces invoke the DEC mechanism together with the base GEMV kernels. The DEC and base GEMV kernels are launched on separate streams and synchronized in order to coerce parallel execution.

### 5. Dequantization

```python
decdec_ext.anyprec_dequant(qweight, lut, bitwidth) -> Tensor
```

Dequantizes the weights in the ANYPREC format back to floating-point format.

For other quantization formats, we have implemented PyTorch-based dequantization functions (TODO: add links to these functions).

### 6. Kernels for Testing

```python
decdec_ext.dummy_anyprec(
    input, output, qweight, lut,
    bitwidth, dummy_sm, dummy_iters
)

decdec_ext.dummy_lutgemm(
    input, output, qweight, alpha, qbias,
    bitwidth, group_size, dummy_sm, dummy_iters
)

decdec_ext.dummy_sqllm(
    input, output, qweight, lut,
    bitwidth, dummy_sm, dummy_iters
)
```

These kernels are intended for testing and benchmarking only. They are not used in actual DecDEC deployment. They can be used to run the ANYPREC, LUTGEMM, or SQLLM GEMV kernels alongside a dummy kernel that occupies SMs, allowing evaluation of performance under constrained SM availability.

## Notes

- This extension requires a CUDA-enabled PyTorch environment.
- All pointer-type arguments (e.g., `dec_config_ptr`) should be treated as integer handles.
- Ensure all input/output tensors are CUDA tensors allocated on the same device.
- Basic sanity checks and tests for a subset of functionalities are included in the `./tests/` directory.
- While we only provide integration on SqueezeLLM and AWQ quantization formats, nothing inherently prevents the integration of other quantization formats and kernels. If you wish to modify the extension, we suggest you start by looking at the `decdec.cu` file, which contains the main logic for integrating DEC with the base GEMV kernels.

## Reference Implementation

For a reference implementation of how to use this extension to deploy DecDEC on an end-to-end LLM inference pipeline, see [end-to-end](../end-to-end/).

## Just show me the code!

Refer to `fused_dec_kernel` in `dec.cu` for the core implementation of the DEC kernel, and `decdec.cu` for how it is integrated with the base GEMV kernels.
