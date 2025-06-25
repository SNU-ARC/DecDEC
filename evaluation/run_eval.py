from evaltools import run_eval, run_eval_ppl, run_eval_fp16, run_eval_ppl_fp16, append_mixed_precision_configs

# Base model identifier (must match the full-precision model)
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Path to quantized model in current directory (should match this format)
ALGO = "anyprec"
BITWIDTH = 3
MODEL_PATHS = {
    3:"./anyprec-(Meta-Llama-3-8B-Instruct)-w3_orig3-gc1-c4_s100_blk512",
    4:"./anyprec-(Meta-Llama-3-8B-Instruct)-w4_orig4-gc1-c4_s100_blk512",
}

# Evaluation configs (name, bitwidth, [xyzw])
# xyzw is the DecDEC parameter set: [n_tb, k_chunk_qkv, k_chunk_ffn1, k_chunk_ffn2]
CONFIGS = [
    ("k_chunk = 0", BITWIDTH, [0, 0, 0, 0]),  # Baseline, no DecDEC
    ("k_chunk = 16", BITWIDTH, [16, 16, 16, 16]),
    ("k_chunk = 32", BITWIDTH, [32, 32, 32, 32]),
    ("k_chunk = 64", BITWIDTH, [64, 64, 64, 64]),
]

# LM Evaluation tasks (use lm-eval-harness task names)
TASKS = [
    "bbh_cot_fewshot_causal_judgement"
]

# Use only wikitext2 for perplexity evaluation
DATASETS = [
    "wikitext2",
]

if __name__ == "__main__":
    # Full-precision baseline (optional)
    print("Running full-precision perplexity eval (fp16)...")
    run_eval_ppl_fp16(BASE_MODEL, DATASETS)

    print("Running full-precision LM eval (fp16)...")
    run_eval_fp16(BASE_MODEL, TASKS)

    # Quantized model: Perplexity Evaluation
    print("Running perplexity eval on quantized model...")
    run_eval_ppl(
        base_model_repo=BASE_MODEL,
        model_paths=MODEL_PATHS,
        algorithms=[ALGO],
        datasets=DATASETS,
        configs=CONFIGS,
        forward_modes=["ppl"],  # For perplexity evaluation, use the "ppl" forward mode
    )

    # Quantized model: LM Evaluation
    print("Running LM eval on quantized model...")
    run_eval(
        base_model_repo=BASE_MODEL,
        model_paths=MODEL_PATHS,
        algorithms=[ALGO],
        tasks=TASKS,
        configs=CONFIGS,
        forward_modes=["decdec"],  # This will use DecDEC for inference
    )

