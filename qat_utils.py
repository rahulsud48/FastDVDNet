"""
QAT helpers — keep the quantization plumbing out of the train/test scripts.

Standard eager-mode QAT lifecycle for this model:
  build(train_mode=True) -> eval() -> fuse_model() -> train() -> set_qconfig()
  -> prepare_qat() -> [train loop with fake-quant] -> (deploy) convert()

Notes:
  * fuse_model() MUST run in eval() (torch requirement).
  * prepare_qat() MUST run in train().
  * convert() runs on a deep-copied eval() model for INT8 inference.
  * During QAT training everything is still FP32 (fake-quant rounds to the INT8
    grid and back); real INT8 only appears after convert().
"""
import copy
import torch
from torch.ao.quantization import prepare_qat, convert


def prepare_model_qat(model):
    """In-place: fuse -> set qconfig -> prepare_qat. Returns the prepared model
    (still FP32 math, with fake-quant observers inserted)."""
    model.eval()
    model.fuse_model()
    model.train()
    model.set_qconfig()
    prepare_qat(model, inplace=True)
    return model


def convert_to_int8(model):
    """Deep-copy, eval, convert to a true-INT8 model for inference/eval.
    Leaves the original QAT model untouched (so training can continue)."""
    m = copy.deepcopy(model).cpu().eval()
    convert(m, inplace=True)
    return m
