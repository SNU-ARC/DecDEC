import os
import re
import warnings
from typing import Dict, Tuple

import torch

def lutgemm_load(
    ckpt_dir: str,
    *,
    map_location: str | torch.device | None = "cpu",
    fp_dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    warnings.filterwarnings("ignore", category=FutureWarning)

    # ------------------------------------------------------------------
    # 1.  Load & rename the model weights
    # ------------------------------------------------------------------
    replacements: dict[str, str] = {
        "embed_tokens": "tok_embeddings",
        "self_attn":    "attention",
        "o_proj":       "wo",
        "qkv_proj":     "wqkv",
        "mlp":          "feed_forward",
        "down_proj":    "w2",
        "gate_up_proj": "w1w3",
        "lm_head":      "output",
        "binary":       "qweight",
    }

    raw = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"),
                     map_location=map_location)

    model_dict: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        new_key = k.replace("model.", "")
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        model_dict[new_key] = v
    del raw                                                   # free memory

    # ------------------------------------------------------------------
    # 2. figure out layer count
    # ------------------------------------------------------------------
    num_layers = max(
        int(m.group(1)) + 1
        for m in (re.search(r"layers\.(\d+)\.", k) for k in model_dict)
        if m
    )

    # ------------------------------------------------------------------
    # 3.  Fuse QKV  &  gate-up projections layer-by-layer
    # ------------------------------------------------------------------
    for i in range(num_layers):
        # ---- QKV -----------------------------------------------------
        qkw = f"layers.{i}.attention.q_proj.qweight"
        kkw = f"layers.{i}.attention.k_proj.qweight"
        vkw = f"layers.{i}.attention.v_proj.qweight"
        if qkw in model_dict:                                   # not yet fused
            fused = torch.cat(
                (model_dict[qkw], model_dict[kkw], model_dict[vkw]), dim=2
            )
            model_dict[f"layers.{i}.attention.wqkv.qweight"] = fused
            for key in (qkw, kkw, vkw):
                del model_dict[key]

            # alpha
            qall = f"layers.{i}.attention.q_proj.alpha"
            kall = f"layers.{i}.attention.k_proj.alpha"
            vall = f"layers.{i}.attention.v_proj.alpha"
            fused = torch.cat(
                (model_dict[qall], model_dict[kall], model_dict[vall]), dim=2
            )
            model_dict[f"layers.{i}.attention.wqkv.alpha"] = fused
            for key in (qall, kall, vall):
                del model_dict[key]

            # q_bias
            qbb = f"layers.{i}.attention.q_proj.q_bias"
            kbb = f"layers.{i}.attention.k_proj.q_bias"
            vbb = f"layers.{i}.attention.v_proj.q_bias"
            fused = torch.cat(
                (model_dict[qbb], model_dict[kbb], model_dict[vbb]), dim=1
            )
            model_dict[f"layers.{i}.attention.wqkv.q_bias"] = fused
            for key in (qbb, kbb, vbb):
                del model_dict[key]

        # ---- Gate-Up (MLP) ------------------------------------------
        gkw = f"layers.{i}.feed_forward.gate_proj.qweight"
        ukw = f"layers.{i}.feed_forward.up_proj.qweight"
        if gkw in model_dict:                                   # not yet fused
            fused = torch.cat((model_dict[gkw], model_dict[ukw]), dim=2)
            model_dict[f"layers.{i}.feed_forward.w1w3.qweight"] = fused
            for key in (gkw, ukw):
                del model_dict[key]

            gall = f"layers.{i}.feed_forward.gate_proj.alpha"
            uall = f"layers.{i}.feed_forward.up_proj.alpha"
            fused = torch.cat((model_dict[gall], model_dict[uall]), dim=2)
            model_dict[f"layers.{i}.feed_forward.w1w3.alpha"] = fused
            for key in (gall, uall):
                del model_dict[key]

            gbb = f"layers.{i}.feed_forward.gate_proj.q_bias"
            ubb = f"layers.{i}.feed_forward.up_proj.q_bias"
            fused = torch.cat((model_dict[gbb], model_dict[ubb]), dim=1)
            model_dict[f"layers.{i}.feed_forward.w1w3.q_bias"] = fused
            for key in (gbb, ubb):
                del model_dict[key]

    # ------------------------------------------------------------------
    # 4.  Build the cheatsheet dict (q_residuals, scales, thresholds)
    # ------------------------------------------------------------------
    cheatsheet_layers = torch.load(os.path.join(ckpt_dir, "cheatsheet.pt"),
                                   map_location=map_location)
    cheatsheet_dict: dict[str, torch.Tensor] = {}

    for i, layer in enumerate(cheatsheet_layers):
        # QKV (cheats & scales)
        if "self_attn.qkv_proj" not in layer:
            qkv_q_residual = torch.cat(
                (
                    layer["self_attn.q_proj"]["cheatsheet"],
                    layer["self_attn.k_proj"]["cheatsheet"],
                    layer["self_attn.v_proj"]["cheatsheet"],
                ),
                dim=1,
            )
            qkv_scales = torch.cat(
                (
                    layer["self_attn.q_proj"]["reordered_scales"],
                    layer["self_attn.k_proj"]["reordered_scales"],
                    layer["self_attn.v_proj"]["reordered_scales"],
                ),
                dim=0,
            )
        else:
            qkv_q_residual = layer["self_attn.qkv_proj"]["cheatsheet"]
            qkv_scales = layer["self_attn.qkv_proj"]["reordered_scales"]

        cheatsheet_dict[f"layers.{i}.attention.wqkv.q_residual"] = qkv_q_residual
        cheatsheet_dict[f"layers.{i}.attention.wqkv.scales"] = qkv_scales
        cheatsheet_dict[f"layers.{i}.attention.wo.q_residual"] = layer[
            "self_attn.o_proj"
        ]["cheatsheet"]
        cheatsheet_dict[f"layers.{i}.attention.wo.scales"] = layer[
            "self_attn.o_proj"
        ]["reordered_scales"]

        # Gate-Up (cheats & scales)
        if "mlp.gate_up_proj" not in layer:
            w1w3_q_residual = torch.cat(
                (
                    layer["mlp.gate_proj"]["cheatsheet"],
                    layer["mlp.up_proj"]["cheatsheet"],
                ),
                dim=1,
            )
            w1w3_scales = torch.cat(
                (
                    layer["mlp.gate_proj"]["reordered_scales"],
                    layer["mlp.up_proj"]["reordered_scales"],
                ),
                dim=0,
            )
        else:
            w1w3_q_residual = layer["mlp.gate_up_proj"]["cheatsheet"]
            w1w3_scales = layer["mlp.gate_up_proj"]["reordered_scales"]

        cheatsheet_dict[f"layers.{i}.feed_forward.w1w3.q_residual"] = w1w3_q_residual
        cheatsheet_dict[f"layers.{i}.feed_forward.w1w3.scales"] = w1w3_scales

    del cheatsheet_layers                                         # free memory

    # thresholds ------------------------------------------------------
    thresholds_layers = torch.load(
        os.path.join(ckpt_dir, "thresholds.pt"), map_location=map_location
    )
    for i, layer in enumerate(thresholds_layers):
        cheatsheet_dict[
            f"layers.{i}.attention.wqkv.thresholds"
        ] = layer["self_attn.qkv_proj"] if "self_attn.qkv_proj" in layer else layer[
            "self_attn.q_proj"
        ]
        cheatsheet_dict[f"layers.{i}.attention.wo.thresholds"] = layer["self_attn.o_proj"]
        cheatsheet_dict[
            f"layers.{i}.feed_forward.w1w3.thresholds"
        ] = layer["mlp.gate_up_proj"] if "mlp.gate_up_proj" in layer else layer[
            "mlp.gate_proj"
        ]
        cheatsheet_dict[f"layers.{i}.feed_forward.w2.thresholds"] = layer["mlp.down_proj"]

    # ------------------------------------------------------------------
    # 5. final dtype cast
    # ------------------------------------------------------------------
    for d in (model_dict, cheatsheet_dict):
        for k, t in list(d.items()):
            if t.dtype in (torch.float32, torch.bfloat16, torch.float16):
                d[k] = t.to(fp_dtype)

    return model_dict, cheatsheet_dict


def ap_load(
    ckpt_dir: str,
    *,
    map_location: str | torch.device | None = "cpu",
    fp_dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    warnings.filterwarnings("ignore", category=FutureWarning)

    # ------------------------------------------------------------------
    # 1. load & rename
    # ------------------------------------------------------------------
    replacements = {
        "embed_tokens": "tok_embeddings",
        "self_attn": "attention",
        "o_proj": "wo",
        "qkv_proj": "wqkv",
        "mlp": "feed_forward",
        "down_proj": "w2",
        "gate_up_proj": "w1w3",
        "lm_head": "output",
    }

    raw = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"),
                     map_location=map_location)

    model_dict: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        k = k.replace("model.", "")
        for old, new in replacements.items():
            k = k.replace(old, new)
        # lut2/3/4  → lut
        if "lut" in k:
            k = re.sub(r"(?<=lut)[234]", "", k)
        model_dict[k] = v
    del raw

    # Make sure all qweight tensors are contiguous
    for name, tensor in list(model_dict.items()):
        if "qweight" in name:
            model_dict[name] = model_dict[name].contiguous()

    # ------------------------------------------------------------------
    # 2. figure out layer count
    # ------------------------------------------------------------------
    num_layers = max(
        int(m.group(1)) + 1
        for m in (re.search(r"layers\.(\d+)\.", k) for k in model_dict)
        if m
    )

    # ------------------------------------------------------------------
    # 3. fuse QKV / Gate-Up
    # ------------------------------------------------------------------
    for i in range(num_layers):
        # QKV ----------------------------------------------------------
        qkw = f"layers.{i}.attention.q_proj.qweight"
        if qkw in model_dict:                                  # not fused yet
            kkw = f"layers.{i}.attention.k_proj.qweight"
            vkw = f"layers.{i}.attention.v_proj.qweight"
            model_dict[f"layers.{i}.attention.wqkv.qweight"] = torch.cat(
                (model_dict[qkw], model_dict[kkw], model_dict[vkw]), dim=1
            )
            for k in (qkw, kkw, vkw):
                del model_dict[k]

            qlut = f"layers.{i}.attention.q_proj.lut"
            klut = f"layers.{i}.attention.k_proj.lut"
            vlut = f"layers.{i}.attention.v_proj.lut"
            model_dict[f"layers.{i}.attention.wqkv.lut"] = torch.cat(
                (model_dict[qlut], model_dict[klut], model_dict[vlut]), dim=0
            )
            for k in (qlut, klut, vlut):
                del model_dict[k]

        # Gate-Up -----------------------------------------------------
        gkw = f"layers.{i}.feed_forward.gate_proj.qweight"
        if gkw in model_dict:
            ukw = f"layers.{i}.feed_forward.up_proj.qweight"
            model_dict[f"layers.{i}.feed_forward.w1w3.qweight"] = torch.cat(
                (model_dict[gkw], model_dict[ukw]), dim=1
            )
            for k in (gkw, ukw):
                del model_dict[k]

            glut = f"layers.{i}.feed_forward.gate_proj.lut"
            ulut = f"layers.{i}.feed_forward.up_proj.lut"
            model_dict[f"layers.{i}.feed_forward.w1w3.lut"] = torch.cat(
                (model_dict[glut], model_dict[ulut]), dim=0
            )
            for k in (glut, ulut):
                del model_dict[k]

    # ------------------------------------------------------------------
    # 4. cheatsheet (q_residuals, scales, thresholds)
    # ------------------------------------------------------------------
    cheatsheet_dict: Dict[str, torch.Tensor] = {}

    # 4-A  cheatsheet_dict + scales --------------------------------------------
    cheatsheet_dict_layers = torch.load(os.path.join(ckpt_dir, "cheatsheet.pt"),
                               map_location=map_location)

    for i, layer in enumerate(cheatsheet_dict_layers):
        # QKV
        if "self_attn.qkv_proj" not in layer:
            qkv_qres = torch.cat(
                (
                    layer["self_attn.q_proj"]["cheatsheet"],
                    layer["self_attn.k_proj"]["cheatsheet"],
                    layer["self_attn.v_proj"]["cheatsheet"],
                ),
                dim=1,
            )
            qkv_scales = torch.cat(
                (
                    layer["self_attn.q_proj"]["reordered_scales"],
                    layer["self_attn.k_proj"]["reordered_scales"],
                    layer["self_attn.v_proj"]["reordered_scales"],
                ),
                dim=0,
            )
        else:
            qkv_qres = layer["self_attn.qkv_proj"]["cheatsheet"]
            qkv_scales = layer["self_attn.qkv_proj"]["reordered_scales"]

        cheatsheet_dict[f"layers.{i}.attention.wqkv.q_residual"] = qkv_qres
        cheatsheet_dict[f"layers.{i}.attention.wqkv.scales"] = qkv_scales
        cheatsheet_dict[f"layers.{i}.attention.wo.q_residual"] = layer["self_attn.o_proj"][
            "cheatsheet"
        ]
        cheatsheet_dict[f"layers.{i}.attention.wo.scales"] = layer["self_attn.o_proj"][
            "reordered_scales"
        ]

        # Gate-Up
        if "mlp.gate_up_proj" not in layer:
            w1w3_qres = torch.cat(
                (
                    layer["mlp.gate_proj"]["cheatsheet"],
                    layer["mlp.up_proj"]["cheatsheet"],
                ),
                dim=1,
            )
            w1w3_scales = torch.cat(
                (
                    layer["mlp.gate_proj"]["reordered_scales"],
                    layer["mlp.up_proj"]["reordered_scales"],
                ),
                dim=0,
            )
        else:
            w1w3_qres = layer["mlp.gate_up_proj"]["cheatsheet"]
            w1w3_scales = layer["mlp.gate_up_proj"]["reordered_scales"]

        cheatsheet_dict[f"layers.{i}.feed_forward.w1w3.q_residual"] = w1w3_qres
        cheatsheet_dict[f"layers.{i}.feed_forward.w1w3.scales"] = w1w3_scales

        # w2 (down-proj) always present
        cheatsheet_dict[f"layers.{i}.feed_forward.w2.q_residual"] = layer["mlp.down_proj"][
            "cheatsheet"
        ]
        cheatsheet_dict[f"layers.{i}.feed_forward.w2.scales"] = layer["mlp.down_proj"][
            "reordered_scales"
        ]

    # 4-B  thresholds --------------------------------------------------
    thresh_layers = torch.load(os.path.join(ckpt_dir, "thresholds.pt"),
                               map_location=map_location)

    for i, layer in enumerate(thresh_layers):
        cheatsheet_dict[f"layers.{i}.attention.wqkv.thresholds"] = (
            layer["self_attn.qkv_proj"]
            if "self_attn.qkv_proj" in layer
            else layer["self_attn.q_proj"]
        )
        cheatsheet_dict[f"layers.{i}.attention.wo.thresholds"] = layer["self_attn.o_proj"]
        cheatsheet_dict[f"layers.{i}.feed_forward.w1w3.thresholds"] = (
            layer["mlp.gate_up_proj"]
            if "mlp.gate_up_proj" in layer
            else layer["mlp.gate_proj"]
        )
        cheatsheet_dict[f"layers.{i}.feed_forward.w2.thresholds"] = layer["mlp.down_proj"]

    # ------------------------------------------------------------------
    # 5. final dtype cast
    # ------------------------------------------------------------------
    for d in (model_dict, cheatsheet_dict):
        for k, t in list(d.items()):
            if t.dtype in (torch.float32, torch.bfloat16, torch.float16):
                d[k] = t.to(fp_dtype)

    return model_dict, cheatsheet_dict