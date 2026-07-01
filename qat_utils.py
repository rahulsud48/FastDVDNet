"""
QAT lifecycle helpers (keep quant plumbing out of train/test).

Order matters:
  build(train_mode) -> eval() -> fuse_model() -> train() -> set_qconfig()
  -> prepare_qat()  -> [train with fake-quant] -> (deploy) convert()

During QAT training everything is FP32 (fake-quant rounds to the INT8 grid and
back); real INT8 appears only after convert().
"""
import copy
import torch
from torch.ao.quantization import prepare_qat, convert


def prepare_model_qat(model):
    """In-place fuse -> qconfig -> prepare_qat. Returns the prepared model."""
    model.eval()
    model.fuse_model()
    model.train()
    model.set_qconfig()
    prepare_qat(model, inplace=True)
    return model


def convert_to_int8(model):
    """Deep-copy, eval, convert to a true-INT8 model (CPU). Original untouched."""
    m = copy.deepcopy(model).cpu().eval()
    convert(m, inplace=True)
    return m
