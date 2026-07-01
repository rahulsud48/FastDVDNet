#!/usr/bin/env python3
"""
convert_qat.py — bridge from a trained QAT checkpoint to the production INT8
inference graph (the models_qat.py form: INT8 in, INT8 out, INT8 bank).

What it does:
  1. Rebuild the QAT model structure (train_mode controls the bottleneck pool).
  2. Load the trained QAT checkpoint (the learned quant scales ride along in the
     state_dict as buffers — including the LEARNED BANK SCALE).
  3. convert() -> a true-INT8 model. The learned scales are baked in automatically.
  4. Read out the learned bank scale/zero-point and save it (bank_scale.txt +
     printed), so the ISP host code knows what fixed scale to keep the rolling
     INT8 bank at in SRAM.
  5. Save the converted INT8 model (state_dict) for inference.

Usage:
  python convert_qat.py --qat_ckpt logs/net_best.pth --out_dir logs/int8 \
      [--train_mode_infer]   # use inference pool (270x480) instead of train pool

Note: the converted model runs on CPU (PyTorch quantized backend is CPU-only).
"""
import os
import argparse
import copy
import torch

from models import FastDVDnet
from utils import remove_dataparallel_wrapper
from qat_utils import prepare_model_qat, convert_to_int8


def main(**a):
    os.makedirs(a['out_dir'], exist_ok=True)

    # train_mode for the rebuilt model: at inference you typically want the
    # static inference pool (train_mode=False). Use --train_mode_infer to force
    # the training pool instead (e.g. to validate at 96x96).
    train_mode = a['train_mode_infer']

    print("> Rebuilding QAT model structure ...")
    model = FastDVDnet(bank_size=a['bank_size'], num_heads=a['num_heads'],
                       pool_size=a['pool_size'], train_mode=train_mode,
                       input_bits=a['input_bits'])
    model = prepare_model_qat(model)   # fuse -> qconfig -> prepare_qat

    print(f"> Loading QAT checkpoint: {a['qat_ckpt']}")
    sd = torch.load(a['qat_ckpt'], map_location='cpu')
    if any(k.startswith('module.') for k in sd):
        sd = remove_dataparallel_wrapper(sd)
    model.load_state_dict(sd)

    # ---- read the LEARNED bank scale BEFORE convert (from the fake-quant) ----
    def _qparams(stub):
        # The learned scale lives on the fake-quant. Depending on torch version
        # the stub IS the fake-quant (has calculate_qparams) or wraps it in
        # .activation_post_process. Handle both.
        if hasattr(stub, 'calculate_qparams'):
            return stub.calculate_qparams()
        return stub.activation_post_process.calculate_qparams()
    bk_stub = model.temp.kv_attn.quant_bank_k
    bv_stub = model.temp.kv_attn.quant_bank_v
    bk_scale, bk_zp = _qparams(bk_stub)
    bv_scale, bv_zp = _qparams(bv_stub)
    bank_scale = float(bk_scale.item())
    bank_zp    = int(bk_zp.item())
    print(f"> Learned bank scale (k): scale={bank_scale:.8f} zero_point={bank_zp}")
    print(f"  (bank scale (v): scale={float(bv_scale.item()):.8f} "
          f"zp={int(bv_zp.item())})")

    # Save the bank scale for the ISP host code.
    scale_path = os.path.join(a['out_dir'], 'bank_scale.txt')
    with open(scale_path, 'w') as f:
        f.write(f"bank_k_scale {bank_scale:.10f}\n")
        f.write(f"bank_k_zero_point {bank_zp}\n")
        f.write(f"bank_v_scale {float(bv_scale.item()):.10f}\n")
        f.write(f"bank_v_zero_point {int(bv_zp.item())}\n")
    print(f"> Saved bank scale -> {scale_path}")

    # ---- convert to INT8 (bakes in all learned scales, incl. bank) ----
    print("> Converting to INT8 ...")
    int8 = convert_to_int8(model)

    ckpt_path = os.path.join(a['out_dir'], 'net_int8.pth')
    torch.save(int8.state_dict(), ckpt_path)
    print(f"> Saved INT8 model -> {ckpt_path}")
    print("\nDone. For inference, load net_int8.pth and feed INT8 (quint8) input;\n"
          "maintain the rolling INT8 bank at the bank scale above.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert QAT checkpoint -> INT8 inference graph")
    p.add_argument("--qat_ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./int8_out")
    p.add_argument("--bank_size", type=int, default=10)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--pool_size", type=int, default=8)
    p.add_argument("--input_bits", type=int, default=8)
    p.add_argument("--train_mode_infer", action='store_true',
                   help="Rebuild with training pool (96x96) instead of inference pool")
    a = p.parse_args()
    main(**vars(a))
