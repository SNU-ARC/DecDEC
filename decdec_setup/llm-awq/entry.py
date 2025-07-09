from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os

from quantize.pre_quant import run_awq, apply_awq
from quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")

parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)

# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights

parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--dump_scaled", type=str, default=None, help="save awq scaled model")
parser.add_argument("--dump_fakequant", type=str, default=None, help="save fake-quantized model")

# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)

args = parser.parse_args()

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path):

    print(f"* Building model {model_path}")


    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
    config.use_cache = False

    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
    # Init model on CPU:
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs
    )
    model.eval()

    if args.run_awq:
        assert args.dump_awq, "Please save the awq results with --dump_awq"

        awq_results = run_awq(
            model,
            enc,
            bit_pt=args.w_bit,
            q_config=q_config,
            n_samples=128,
            seqlen=512,
        )
        if args.dump_awq:
            dirpath = os.path.dirname(args.dump_awq)
            os.makedirs(dirpath, exist_ok=True)

            torch.save(awq_results, args.dump_awq)
            print("AWQ results saved at", args.dump_awq)

        exit(0)

    if args.load_awq:
        print("Loading pre-computed AWQ results from", args.load_awq)
        awq_results = torch.load(args.load_awq, map_location="cpu")
        apply_awq(model, awq_results,args)
        if args.dump_scaled:
            model.save_pretrained(args.dump_scaled)
            exit(0)
    # weight quantization
    if args.w_bit is not None:
        if args.q_backend == "fake":
            assert (
                args.dump_quant is None
            ), "Need to use real quantization to dump quantized weights"
            pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
            if args.dump_fakequant:
                model.save_pretrained(args.dump_fakequant)
                print("Pseudo-quantized models saved at", args.dump_fakequant)
        elif args.q_backend == "real":  # real quantization
            real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
            if args.dump_quant:
                dirpath = os.path.dirname(args.dump_quant)
                os.makedirs(dirpath, exist_ok=True)

                print(f"Saving the quantized model at {args.dump_quant}...")

                filtered_state_dict = {k: v for k, v in model.cpu().state_dict().items() if not any(substr in k for substr in ['qweight', 'scaled_zeros', 'scales'])}

                torch.save(filtered_state_dict,args.dump_quant)
                exit(0)
        else:
            raise NotImplementedError

    return model, enc


def main():

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)

if __name__ == "__main__":
    main()
