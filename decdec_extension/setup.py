import os
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="decdec_ext",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="decdec_ext", 
            sources=["bindings.cpp", "dec_context.cu", "dec_config.cu",
                    "decdec.cu", "dec.cu", "sqllm.cu", "lutgemm.cu", "anyprec.cu"],
            extra_compile_args={
                'cxx': ["-O3", "-DENABLE_BF16"],
                'nvcc': [
                    '-lineinfo', 
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_HALF2_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    # Add or remove architectures as needed
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                    "-gencode=arch=compute_120,code=sm_120",
                ]
            },
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
)
