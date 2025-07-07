# End-to-End LLM Inference with DecDEC

1. Convert quantized checkpoint
      ```bash
      # SqueezeLLM models (Any-precision LLM backend)
      python ap_convert.py --ckpt_dir ${CHECKPOINT_PATH}
      
      # AWQ models (LUTGEMM LLM backend)
      python lutgemm_convert.py --ckpt_dir ${CHECKPOINT_PATH}
      ```
      
2. Convert quantized checkpoint
      ```bash
      # For decoding speed benchmarking
      python generate.py --compile 2 --num_samples 5 \
            --model_name ${MODEL_NAME} --bitwidth ${BITWIDTH} --dtype "float16" \
            --checkpoint_path ${CHECKPOINT_PATH} --backend ${BACKEND} \
            --n_tbs ${N_TBS} --k_chunks ${K_CHUNK_LIST} \
	    --max_new_tokens 100

      # For text generation
      python generate.py --prompt "YOUR PROOMPT HERE" --compile 2 --num_samples 1 \
            --model_name ${MODEL_NAME} --bitwidth ${BITWIDTH} --dtype "float16" \
            --checkpoint_path ${CHECKPOINT_PATH} --backend ${BACKEND} \
            --n_tbs ${N_TBS} --k_chunks ${K_CHUNK_LIST} \
	    --max_new_tokens 100 --print_result
      ```



