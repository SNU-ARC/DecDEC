MODEL=microsoft/Phi-3-medium-4k-instruct
MODEL_NAME=phi-medium-4k-instruct
BITS=4

# run AWQ search (optional; we provided the pre-computed results)
CUDA_VISIBLE_DEVICES=0 python -m entry --model_path $MODEL \
    --w_bit $BITS --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL_NAME-w$BITS-g128.pt

# generate scaled awq model
CUDA_VISIBLE_DEVICES=0 python -m entry --model_path $MODEL \
    --w_bit $BITS --q_group_size 128 \
    --load_awq awq_cache/$MODEL_NAME-w$BITS-g128.pt \
    --q_backend fake --dump_scaled scaled_cache/$MODEL_NAME-w$BITS-g128.pt

# generate fake quantized weights
CUDA_VISIBLE_DEVICES=0 python -m entry --model_path $MODEL \
    --w_bit $BITS --q_group_size 128 \
    --load_awq awq_cache/$MODEL_NAME-w$BITS-g128.pt \
    --q_backend fake --dump_fakequant fakequant_cache/$MODEL_NAME-w$BITS-g128.pt

# generate real quantized weights (w3)
CUDA_VISIBLE_DEVICES=0 python -m entry --model_path $MODEL \
    --w_bit $BITS --q_group_size 128 \
    --load_awq awq_cache/$MODEL_NAME-w$BITS-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL_NAME-w$BITS-g128-lutgemm.pt