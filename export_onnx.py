#!/usr/bin/env python3
"""
export_onnx.py — export the QAT model to a PRE-QUANTIZED INT8 ONNX graph
(.onnx) at FIXED shapes, with bank as graph I/O.

HONEST SCOPE (read this):
  * PyTorch's quantized CONV ops export to ONNX as real int8 (QuantizeLinear/
    DequantizeLinear + integer Conv). GOOD.
  * PyTorch's quantized MATMUL (quantized::matmul, used by the KV-bank attention)
    has NO ONNX export support. So the bank attention matmuls + softmax are
    exported as a FLOAT ISLAND (dequant -> float matmul -> softmax -> requant),
    exactly like the softmax island the compiler team already handles.
  * Net result: convs/adds = int8; bank attention = float island wrapped in QDQ.
    This is the only configuration that actually exports today, and the float
    island is NPU-handleable (same pattern as softmax).

Fixed geometry: y (1,1,1080,1920), uv_noise (1,4,270,480), bank (1,640,32).
No dynamic axes (your compiler rejects them).

Usage:
  python export_onnx.py --qat_ckpt logs/net_best.pth --out model_int8.onnx
"""
import argparse
import copy
import torch
import torch.nn.functional as F

import onnx
from models import FastDVDnet, BottleneckCrossAttn
from utils import remove_dataparallel_wrapper
from qat_utils import prepare_model_qat, convert_to_int8


# Fixed inference geometry.
H, W = 1080, 1920
BANK_LEN, BANK_DIM = 640, 32


def _attn_forward_float_island(self, x, bank_k, bank_v):
    """BottleneckCrossAttn.forward with the bank attention as a FLOAT ISLAND so
    the model exports to ONNX (no quantized::matmul). Conv path stays int8.
    Mirrors the original math: qk^T/8 -> softmax -> @v -> gate*x + x.
    Keeps the /8.0 scale."""
    N, C, Hh, Ww = x.shape
    tokens = self._to_tokens(x)
    q_cur = self.to_q(tokens)
    k_cur = self.to_k(tokens)
    v_cur = self.to_v(tokens)

    bk = self.quant_bank_k(bank_k)
    bv = self.quant_bank_v(bank_v)

    # Float island: dequantize q/bk/bv, do float matmul+softmax+matmul, requant.
    qf  = self.deq_pre_softmax(q_cur)
    bkf = self.deq_pre_softmax(bk)
    qkT = torch.matmul(qf, bkf.transpose(-2, -1)) / 8.0
    attn_map = F.softmax(qkT, dim=-1)
    bvf = self.deq_pre_softmax(bv)
    attn_out = torch.matmul(attn_map, bvf)
    attn_out = self.q_post_softmax(attn_out)

    gate = self.gate(attn_out.mean(dim=1)).view(N, C, 1, 1)
    x_g  = self.gate_mul.mul(x, gate)
    x_out = self.gate_add.add(x_g, x)
    return x_out, k_cur, v_cur


def main(**a):
    print("> Rebuilding QAT model (inference pool, train_mode=False) ...")
    # Apply the float-island attention forward so the export succeeds.
    BottleneckCrossAttn.forward = _attn_forward_float_island

    model = FastDVDnet(bank_size=a['bank_size'], num_heads=a['num_heads'],
                       pool_size=a['pool_size'], train_mode=False,
                       input_bits=a['input_bits'])
    model = prepare_model_qat(model)

    print(f"> Loading QAT checkpoint: {a['qat_ckpt']}")
    sd = torch.load(a['qat_ckpt'], map_location='cpu')
    if any(k.startswith('module.') for k in sd):
        sd = remove_dataparallel_wrapper(sd)
    model.load_state_dict(sd)

    print("> Converting to INT8 ...")
    int8 = convert_to_int8(model)

    print("> Exporting ONNX (fixed shapes, bank as I/O) ...")
    y   = torch.rand(1, 1, H, W) * 255
    uvn = torch.rand(1, 4, H // 4, W // 4) * 255
    bk  = torch.zeros(1, BANK_LEN, BANK_DIM)
    bv  = torch.zeros(1, BANK_LEN, BANK_DIM)

    torch.onnx.export(
        int8, (y, uvn, bk, bv), a['out'],
        input_names=['y', 'uv_noise', 'bank_k', 'bank_v'],
        output_names=['y_res', 'uv_res', 'curr_k', 'curr_v'],
        opset_version=a['opset'],
        dynamic_axes=None,          # FIXED shapes — no dynamic axes
        dynamo=False,               # legacy exporter handles quantized convs
    )

    g = onnx.load(a['out'])
    onnx.checker.check_model(g)
    ops = sorted(set(n.op_type for n in g.graph.node))
    has_qdq = ('QuantizeLinear' in ops) and ('DequantizeLinear' in ops)
    print(f"> ONNX written: {a['out']}")
    print(f"  ops: {ops}")
    print(f"  QDQ present (int8 path): {has_qdq}")
    print(f"  inputs : y(1,1,{H},{W}) uv_noise(1,4,{H//4},{W//4}) "
          f"bank_k/v(1,{BANK_LEN},{BANK_DIM})")
    print(f"  outputs: y_res, uv_res, curr_k, curr_v")
    print("\nNote: bank attention matmul + softmax are a FLOAT ISLAND (QDQ-wrapped);"
          "\nconvs are real int8. Hand to the NPU compiler.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export QAT model -> pre-quantized INT8 ONNX")
    p.add_argument("--qat_ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="model_int8.onnx")
    p.add_argument("--bank_size", type=int, default=10)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--pool_size", type=int, default=8)
    p.add_argument("--input_bits", type=int, default=8)
    p.add_argument("--opset", type=int, default=13)
    a = p.parse_args()
    main(**vars(a))
