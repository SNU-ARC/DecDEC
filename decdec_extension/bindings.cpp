#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dec_config.h"

#include "sqllm.h"
#include "lutgemm.h"

#include "decdec.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("create_dec_context", &create_dec_context, "Create DECContext (returns pointer)");
	m.def("create_dec_config", &create_dec_config, "Create DECConfig (returns pointer)");
	m.def("read_dec_config", &read_dec_config, "Read info from a DECConfig pointer");
	m.def("update_dec_config", &update_dec_config, "Update a DECConfig pointer");
	m.def("destroy_dec_config", &destroy_dec_config, "Destroy a DECConfig pointer");

	m.def("dec", &dec, "standalone DEC");
	m.def("unfused_dec", &unfused_dec, "DEC without row selection, for testing purposes");

	m.def("anyprec_dequant", &anyprec_dequant, "ANYPREC dequantization");

	m.def("anyprec_gemv", &anyprec_gemv, "ANYPREC GEMV");
	m.def("lutgemm_gemv", &lutgemm_gemv, "LUTGEMM GEMV");
	m.def("sqllm_gemv", &sqllm_gemv, "SQLLM GEMV");

	m.def("dec_anyprec", &dec_anyprec, "ANYPREC with DEC");
	m.def("dec_lutgemm", &dec_lutgemm, "LUTGEMM with DEC");
	m.def("dec_sqllm", &dec_sqllm, "SQLLM with DEC");

	m.def("dummy_anyprec", &dummy_anyprec, "ANYPREC GEMV with dummy SM blocker");
	m.def("dummy_lutgemm", &dummy_lutgemm, "LUTGEMM GEMV with dummy SM blocker");
	m.def("dummy_sqllm", &dummy_sqllm, "SQLLM GEMV with dummy SM blocker");
}
