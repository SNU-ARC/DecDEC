import torch
import os
import numpy as np
import argparse
import warnings

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

replacements = {
    'embed_tokens': 'tok_embeddings',
    'self_attn': 'attention',
    'o_proj': 'wo',
    'qkv_proj': 'wqkv',
    'mlp': 'feed_forward',
    'down_proj': 'w2',
    'gate_up_proj': 'w1w3',
    'lm_head': 'output',
    'binary': 'qweight'
}

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str)
args = parser.parse_args()

ckpt_dir = args.ckpt_dir
ckpt = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"))

new_dict = {}
for key, value in ckpt.items():
    # Remove 'model.' from the key
    new_key = key.replace('model.', '')

    # Perform the replacements as specified
    for old, new in replacements.items():
        new_key = new_key.replace(old, new)

    # Update the new dictionary
    new_dict[new_key] = value

for key in new_dict.keys():
    if (new_dict[key].dtype == torch.bfloat16):
        new_dict[key] = new_dict[key].half()
    if ('alpha' in key):
        new_dict[key] = new_dict[key].half()
    if ('q_bias' in key):
        new_dict[key] = new_dict[key].half()

for i in range(32):
    # qkv fusion
    if 'wqkv' not in new_dict:
        key_q_qweight = 'layers.'+str(i)+'.attention.q_proj.qweight'
        key_k_qweight = 'layers.'+str(i)+'.attention.k_proj.qweight'
        key_v_qweight = 'layers.'+str(i)+'.attention.v_proj.qweight'
        new_key_qweight = 'layers.'+str(i)+'.attention.wqkv.qweight'

        new_dict[new_key_qweight] = torch.cat((new_dict[key_q_qweight],
                                               new_dict[key_k_qweight],
                                               new_dict[key_v_qweight]), dim=2)

        del(new_dict[key_q_qweight])
        del(new_dict[key_k_qweight])
        del(new_dict[key_v_qweight])


        key_q_lut = 'layers.'+str(i)+'.attention.q_proj.alpha'
        key_k_lut = 'layers.'+str(i)+'.attention.k_proj.alpha'
        key_v_lut = 'layers.'+str(i)+'.attention.v_proj.alpha'
        new_key_lut = 'layers.'+str(i)+'.attention.wqkv.alpha'

        new_dict[new_key_lut] = torch.cat((new_dict[key_q_lut],
                                               new_dict[key_k_lut],
                                               new_dict[key_v_lut]), dim=2)

        del(new_dict[key_q_lut])
        del(new_dict[key_k_lut])
        del(new_dict[key_v_lut])
        
        key_q_q_bias = 'layers.'+str(i)+'.attention.q_proj.q_bias'
        key_k_q_bias = 'layers.'+str(i)+'.attention.k_proj.q_bias'
        key_v_q_bias = 'layers.'+str(i)+'.attention.v_proj.q_bias'
        new_key_q_bias = 'layers.'+str(i)+'.attention.wqkv.q_bias'

        new_dict[new_key_q_bias] = torch.cat((new_dict[key_q_q_bias],
                                               new_dict[key_k_q_bias],
                                               new_dict[key_v_q_bias]), dim=1)

        del(new_dict[key_q_q_bias])
        del(new_dict[key_k_q_bias])
        del(new_dict[key_v_q_bias])
    
    # gate up fusion
    if 'w1w3' not in new_dict:
        key_gate_qweight = 'layers.'+str(i)+'.feed_forward.gate_proj.qweight'
        key_up_qweight = 'layers.'+str(i)+'.feed_forward.up_proj.qweight'
        new_key_qweight = 'layers.'+str(i)+'.feed_forward.w1w3.qweight'
        new_dict[new_key_qweight] = torch.cat((new_dict[key_gate_qweight],
                                               new_dict[key_up_qweight]), dim=2)

        del(new_dict[key_gate_qweight])
        del(new_dict[key_up_qweight])

        key_gate_lut = 'layers.'+str(i)+'.feed_forward.gate_proj.alpha'
        key_up_lut = 'layers.'+str(i)+'.feed_forward.up_proj.alpha'
        new_key_lut = 'layers.'+str(i)+'.feed_forward.w1w3.alpha'
        new_dict[new_key_lut] = torch.cat((new_dict[key_gate_lut],
                                               new_dict[key_up_lut]), dim=2)
        del(new_dict[key_gate_lut])
        del(new_dict[key_up_lut])

        key_gate_q_bias = 'layers.'+str(i)+'.feed_forward.gate_proj.q_bias'
        key_up_q_bias = 'layers.'+str(i)+'.feed_forward.up_proj.q_bias'
        new_key_q_bias = 'layers.'+str(i)+'.feed_forward.w1w3.q_bias'
        new_dict[new_key_q_bias] = torch.cat((new_dict[key_gate_q_bias],
                                               new_dict[key_up_q_bias]), dim=1)
        del(new_dict[key_gate_q_bias])
        del(new_dict[key_up_q_bias])

torch.save(new_dict, os.path.join(ckpt_dir, "converted_pytorch_model.bin"))
del(ckpt)
del(new_dict)

###### cheatsheet #######
ckpt = torch.load(os.path.join(ckpt_dir, "cheatsheet.pt"))

new_dict = {}
for i, layer in enumerate(ckpt):
    new_dict['layers.'+str(i)+'.attention.wqkv.q_residual'] = layer["self_attn.qkv_proj"]["cheatsheet"]
    new_dict['layers.'+str(i)+'.attention.wqkv.scales'] = layer["self_attn.qkv_proj"]["reordered_scales"].half()
    new_dict['layers.'+str(i)+'.attention.wo.q_residual'] = layer["self_attn.o_proj"]["cheatsheet"]
    new_dict['layers.'+str(i)+'.attention.wo.scales'] = layer["self_attn.o_proj"]["reordered_scales"].half()

    new_dict['layers.'+str(i)+'.feed_forward.w1w3.q_residual'] = layer["mlp.gate_up_proj"]["cheatsheet"]
    new_dict['layers.'+str(i)+'.feed_forward.w1w3.scales'] = layer["mlp.gate_up_proj"]["reordered_scales"].half()
    new_dict['layers.'+str(i)+'.feed_forward.w2.q_residual'] = layer["mlp.down_proj"]["cheatsheet"]
    new_dict['layers.'+str(i)+'.feed_forward.w2.scales'] = layer["mlp.down_proj"]["reordered_scales"].half()
    
del(ckpt)

###### threshold #######
ckpt = torch.load(os.path.join(ckpt_dir, "thresholds.pt"))

for i, layer in enumerate(ckpt):
    new_dict['layers.'+str(i)+'.attention.wqkv.thresholds'] = layer["self_attn.qkv_proj"].half()
    new_dict['layers.'+str(i)+'.attention.wo.thresholds'] = layer["self_attn.o_proj"].half()

    new_dict['layers.'+str(i)+'.feed_forward.w1w3.thresholds'] = layer["mlp.gate_up_proj"].half()
    new_dict['layers.'+str(i)+'.feed_forward.w2.thresholds'] = layer["mlp.down_proj"].half()

torch.save(new_dict, os.path.join(ckpt_dir, "cheatsheet.bin"))

del(ckpt)
del(new_dict)


