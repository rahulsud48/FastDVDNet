#!/usr/bin/env python3
"""
export_qat_onnx.py — export the QAT FastDVDnet graph to ONNX.

Two modes:
  --mode fakequant  (default)
      Export the PREPARED (prepare_qat) model, i.e. the FakeQuantize graph.
      torch.onnx.export emits QuantizeLinear/DequantizeLinear (QDQ) node pairs
      that mirror each fake-quant. Tensors are still fp32-simulated; this is the
      "training graph" view and is what most int8 toolchains (TensorRT, ORT)
      actually want to ingest for a QAT model.

  --mode int8
      Run convert() first, then export the true-INT8 CPU graph. PyTorch's
      quantized-ONNX export is limited; use this only if your downstream
      toolchain specifically wants the converted form.

IMPORTANT — patch required:
  models.py -> DenBlock.forward contains debug print(...) calls and a sys.exit()
  right before `return`. Those abort ANY forward pass (train/infer/trace). This
  script monkeypatches DenBlock.forward at import time to a clean version so the
  trace can complete. It does NOT modify your file on disk.

Usage:
  # random-initialised weights (structure-only export, no checkpoint needed)
  python export_qat_onnx.py --mode fakequant --out fastdvdnet_qat.onnx

  # from a trained QAT checkpoint
  python export_qat_onnx.py --mode fakequant --qat_ckpt logs/net_best.pth \
      --out fastdvdnet_qat.onnx

  # inference bottleneck pool (270x480 static) instead of training pool
  python export_qat_onnx.py --infer_pool ...

Inputs to the graph (YUV path, matches DenBlock.forward signature):
  y            : (1, 1, H,   W)     full-res noisy Y
  uv_noise     : (1, 4, H/4, W/4)   cat[UV(2), sigma_read(1), lambda_shot(1)]
  bank_k       : (1, 10*64, 32)
  bank_v       : (1, 10*64, 32)
Outputs:
  y_res, uv_res, curr_k, curr_v
"""
import argparse
import sys
import torch
import torch.nn as nn

from models import FastDVDnet, DenBlock
from qat_utils import prepare_model_qat, convert_to_int8
from utils import remove_dataparallel_wrapper


# ---------------------------------------------------------------------------
# Patch: clean DenBlock.forward (strip debug prints + sys.exit so trace runs).
# Body is identical to your models.py version minus the debug lines.
# ---------------------------------------------------------------------------
def _clean_denblock_forward(self, x, uv_noise, bank_k, bank_v):
    x0 = self.input_conv_block(x)
    x1 = self.downsample0(x0)
    x2 = self.downsample1(x1)
    x2 = self.bottleneck_cat.cat((x2, uv_noise), dim=1)
    x2, k_cur, v_cur = self.kv_attn(x2, bank_k, bank_v)
    uv_res = self.output_conv_block_uv(x2)
    x2 = self.upsample2(x2)
    x1 = self.upsample1(self.skip_add1.add(x1, x2))
    x = self.output_conv_block_y(self.skip_add0.add(x0, x1))
    return x, uv_res, k_cur, v_cur


def build_model(args):
    DenBlock.forward = _clean_denblock_forward  # apply patch in-memory

    train_mode = not args.infer_pool  # train pool (96x96) unless --infer_pool
    model = FastDVDnet(
        bank_size=args.bank_size, num_heads=args.num_heads,
        pool_size=args.pool_size, train_mode=train_mode,
        input_bits=args.input_bits,
    )

    if args.mode == "fp32":
        # Plain FP32 export: NO fuse, NO qconfig, NO prepare_qat.
        # In models.py the QuantStub/DeQuantStub are identity ops until
        # prepare_qat runs, so an unprepared FastDVDnet IS the FP32 model.
        # Conv/BN stay as separate ops -> the ONNX has explicit BatchNorm nodes
        # (do_constant_folding will fold them into the convs at export time).
        print("> FP32 mode: skipping fuse / qconfig / prepare_qat.")
    else:
        # Correct order: fuse -> set qconfig -> prepare_qat. We set the
        # EXPORTABLE qconfig BEFORE prepare_qat so plain FakeQuantize modules
        # are instantiated from the start (setting qconfig AFTER prepare_qat is
        # a no-op — the QAT modules already exist and won't be re-swapped).
        from torch.ao.quantization import prepare_qat
        model.eval()
        model.fuse_model()
        model.train()

        if args.export_qconfig:
            _set_exportable_qconfig(model, selective=args.selective_quant,
                                    ff_names=args.quant_ff)
        else:
            model.set_qconfig()              # original 'x86' (NOT exportable)

        prepare_qat(model, inplace=True)

        # Belt-and-suspenders: physically replace any
        # FusedMovingAvgObsFakeQuantize that still slipped through, copying its
        # learned buffers into a plain FakeQuantize. Guarantees the exporter
        # never sees the fused aten op.
        if args.export_qconfig:
            _swap_fused_fakequant(model)

    if args.qat_ckpt:
        kind = "FP32" if args.mode == "fp32" else "QAT"
        print(f"> Loading {kind} checkpoint: {args.qat_ckpt}")
        sd = torch.load(args.qat_ckpt, map_location="cpu")
        if any(k.startswith("module.") for k in sd):
            sd = remove_dataparallel_wrapper(sd)
        # strict=False: buffer sets match, but be lenient across torch versions.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [load] missing keys: {len(missing)} (first: {missing[:3]})")
        if unexpected:
            print(f"  [load] unexpected keys: {len(unexpected)} "
                  f"(first: {unexpected[:3]})")
    else:
        print("> No checkpoint given: exporting structure with current weights.")

    model.cpu().eval()
    return model


def _set_exportable_qconfig(model, selective=True, ff_names=None):
    """Attach a QConfig whose fake-quant is the plain, ONNX-exportable kind.

    'x86' uses FusedMovingAvgObsFakeQuantize -> aten fused_moving_avg_obs_
    fake_quant, which has NO ONNX symbolic at any opset. Plain FakeQuantize ->
    fake_quantize_per_(tensor|channel)_affine, which lowers to QuantizeLinear/
    DequantizeLinear at opset >= 13.

    selective=True: only compute-heavy ops (Conv / Linear / matmul-carrying
    FloatFunctional) get a qconfig. Everything else gets qconfig=None, which
    means prepare_qat inserts NO observer and the export emits NO QDQ pair for
    that module.

    NOTE ON SEMANTICS: eager-mode quantization attaches fake-quant to a
    module's OUTPUT, not its inputs. So "QDQ before Conv" is really "fake-quant
    on whatever feeds the Conv". Disabling quant on a producer (e.g. cat) means
    its consumer receives an unquantized tensor and the graph carries no scale
    for it -- make sure your compiler can re-derive that scale.
    """
    from torch.ao.quantization import (
        FakeQuantize, MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver, QConfig,
    )
    act_fq = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0, quant_max=255,
        dtype=torch.quint8, qscheme=torch.per_tensor_affine,
    )
    wt_fq = FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
    )
    qconfig = QConfig(activation=act_fq, weight=wt_fq)

    if not selective:
        model.qconfig = qconfig
        for m in model.modules():
            m.qconfig = qconfig
        print("> Set exportable qconfig on ALL modules.")
        return

    # ---- selective: start from None everywhere, then enable compute ops ----
    from torch.ao.nn.quantized import FloatFunctional
    import torch.nn.intrinsic as nni

    model.qconfig = None
    for m in model.modules():
        m.qconfig = None

    # Module types that should run in int8 (and therefore need fake-quant).
    QUANT_TYPES = (
        nn.Conv2d, nn.Linear,
        # fused variants produced by fuse_model()
        nni.ConvBn2d, nni.ConvBnReLU2d, nni.ConvReLU2d, nni.LinearReLU,
    )

    # FloatFunctional instances that carry a matmul (GEMM-like) -> quantize.
    # Named on the attention module in models.py. Override via --quant_ff.
    MATMUL_FF_NAMES = set(ff_names) if ff_names else {"attn_qk", "attn_av"}

    n_conv = n_ff = 0
    for name, m in model.named_modules():
        if isinstance(m, QUANT_TYPES):
            m.qconfig = qconfig
            n_conv += 1
        elif isinstance(m, FloatFunctional):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in MATMUL_FF_NAMES:
                m.qconfig = qconfig
                n_ff += 1
            # skip_add0/1, bottleneck_cat, gate_mul, gate_add stay None
    print(f"> Selective qconfig: {n_conv} conv/linear + {n_ff} matmul "
          f"FloatFunctional quantized; all other modules qconfig=None "
          f"(no QDQ emitted).")


def _swap_fused_fakequant(model):
    """Replace surviving FusedMovingAvgObsFakeQuantize with plain FakeQuantize,
    copying learned scale/zero_point/observer buffers across."""
    from torch.ao.quantization import FakeQuantize
    try:
        from torch.ao.quantization.fake_quantize import (
            FusedMovingAvgObsFakeQuantize,
        )
    except ImportError:
        return

    def _clone_to_plain(fused):
        plain = FakeQuantize(
            observer=type(fused.activation_post_process),
            quant_min=fused.quant_min, quant_max=fused.quant_max,
            dtype=fused.dtype, qscheme=fused.qscheme,
        )
        # Copy learned state.
        plain.scale.data = fused.scale.data.clone()
        plain.zero_point.data = fused.zero_point.data.clone()
        src, dst = fused.activation_post_process, plain.activation_post_process
        for buf in ("min_val", "max_val"):
            if hasattr(src, buf) and hasattr(dst, buf):
                getattr(dst, buf).data = getattr(src, buf).data.clone()
        plain.eval()
        return plain

    n = 0
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, FusedMovingAvgObsFakeQuantize):
                setattr(module, name, _clone_to_plain(child))
                n += 1
    print(f"> Swapped {n} fused fake-quant module(s) -> plain FakeQuantize.")


def make_dummy_inputs(args):
    # Spatial size for the trace. Must be a multiple of 4 for the /4 bottleneck.
    # Training pool expects 96x96; inference pool expects 1080x1920.
    if args.infer_pool:
        H, W = 1080, 1920
    else:
        H, W = args.height, args.width
    assert H % 4 == 0 and W % 4 == 0, "H and W must be multiples of 4"

    y = torch.randn(1, 1, H, W)
    uv_noise = torch.randn(1, 4, H // 4, W // 4)
    bank_k = torch.randn(1, 10 * 64, 32)
    bank_v = torch.randn(1, 10 * 64, 32)
    return (y, uv_noise, bank_k, bank_v)


def _simplify_onnx(path):
    """Strip Identity nodes and fold redundant structure.

    Tries onnx-simplifier first (best result: also folds shape subgraphs like
    the Shape/Gather/Unsqueeze/Concat chains from dynamic_axes). Falls back to
    a manual Identity-removal pass if onnxsim isn't installed.
    """
    import onnx

    model = onnx.load(path)
    n_before = len(model.graph.node)

    try:
        import onnxsim
        model, ok = onnxsim.simplify(model)
        if not ok:
            print("  [simplify] onnxsim reported failure; keeping original.")
            return
        onnx.save(model, path)
        print(f"  [simplify] onnxsim: {n_before} -> {len(model.graph.node)} nodes")
        return
    except ImportError:
        print("  [simplify] onnxsim not installed; using manual Identity strip.")
        print("             (pip install onnxsim  for a better result)")

    # ---- manual fallback: rewire and drop Identity nodes ----
    graph = model.graph
    outputs = {o.name for o in graph.output}
    remap = {}
    keep = []
    for node in graph.node:
        # Never drop an Identity that produces a graph output (its name matters).
        if node.op_type == "Identity" and node.output[0] not in outputs:
            remap[node.output[0]] = node.input[0]
        else:
            keep.append(node)

    # Resolve chained identities (a -> b -> c).
    def _resolve(name):
        seen = set()
        while name in remap and name not in seen:
            seen.add(name)
            name = remap[name]
        return name

    for node in keep:
        for i, inp in enumerate(node.input):
            node.input[i] = _resolve(inp)

    del graph.node[:]
    graph.node.extend(keep)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  [simplify] manual: {n_before} -> {len(keep)} nodes "
          f"({n_before - len(keep)} Identity removed)")


def main():
    p = argparse.ArgumentParser(description="Export QAT FastDVDnet -> ONNX")
    p.add_argument("--mode", choices=["fp32", "fakequant", "int8"],
                   default="fakequant",
                   help="fp32: plain float model (no fuse/no quant). "
                        "fakequant: QDQ training graph. int8: converted (limited).")
    p.add_argument("--qat_ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default="fastdvdnet_qat.onnx")
    p.add_argument("--bank_size", type=int, default=10)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--pool_size", type=int, default=8)
    p.add_argument("--input_bits", type=int, default=8)
    p.add_argument("--height", type=int, default=96, help="trace H (train pool)")
    p.add_argument("--width", type=int, default=96, help="trace W (train pool)")
    p.add_argument("--infer_pool", action="store_true",
                   help="Use inference pool (1080x1920) instead of train pool")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--export_qconfig", action="store_true", default=True,
                   help="Swap fused fake-quant for ONNX-exportable FakeQuantize "
                        "(required for fakequant mode; on by default)")
    p.add_argument("--no_export_qconfig", dest="export_qconfig",
                   action="store_false")
    p.add_argument("--selective_quant", action="store_true", default=True,
                   help="Only Conv/Linear/matmul get QDQ; softmax, cat, pool, "
                        "add, upsample get qconfig=None (no QDQ emitted).")
    p.add_argument("--quant_all", dest="selective_quant", action="store_false",
                   help="Quantize every module (original blanket behaviour).")
    p.add_argument("--quant_ff", nargs="*", default=None,
                   help="FloatFunctional attribute names to quantize. Default: "
                        "attn_qk attn_av. Use '--quant_ff attn_av' to drop the "
                        "QDQ between the q.k^T matmul and softmax.")
    p.add_argument("--dynamic", action="store_true", default=False,
                   help="Export with dynamic H/W/bank_len (adds Shape/Gather "
                        "subgraphs). Default is a static graph.")
    p.add_argument("--simplify", action="store_true", default=True,
                   help="Strip Identity nodes / fold shape subgraphs after export")
    p.add_argument("--no_simplify", dest="simplify", action="store_false")
    args = p.parse_args()

    if args.mode == "fakequant" and args.opset < 13:
        print(f"> opset {args.opset} < 13 cannot emit QDQ; bumping to 13.")
        args.opset = 13

    model = build_model(args)

    if args.mode == "int8":
        print("> Converting to true INT8 before export ...")
        model = convert_to_int8(model)
        model.eval()

    dummy = make_dummy_inputs(args)

    # Freeze quant state for export: stop observers from updating (their
    # min_val/max_val.copy_() traces to aten::copy, which has no ONNX symbolic)
    # and lock fake-quant scales. This is also semantically correct for export —
    # we want fixed, learned scales, not live re-estimation.
    # FP32 mode has no fake-quants/observers, so this is skipped entirely.
    if args.mode != "fp32":
        from torch.ao.quantization import disable_observer
        model.apply(disable_observer)
        for m in model.modules():
            # Freeze scale/zero_point (no in-place buffer writes at trace).
            if hasattr(m, "observer_enabled"):
                m.observer_enabled[0] = 0
            if hasattr(m, "fake_quant_enabled"):
                m.fake_quant_enabled[0] = 1  # keep quant active, just frozen
    model.eval()

    # Sanity forward before tracing.
    with torch.no_grad():
        _ = model(*dummy)
    print("> Forward OK, tracing to ONNX ...")

    input_names = ["y", "uv_noise", "bank_k", "bank_v"]
    output_names = ["y_res", "uv_res", "curr_k", "curr_v"]

    # dynamic_axes forces runtime shape arithmetic -> Shape/Gather/Unsqueeze/
    # Concat subgraphs all over the exported graph. For a fixed-resolution NPU
    # target you almost always want a fully STATIC graph instead.
    dyn = None
    if args.dynamic:
        dyn = {
            "y":        {2: "H", 3: "W"},
            "uv_noise": {2: "Hq", 3: "Wq"},
            "bank_k":   {1: "bank_len"},
            "bank_v":   {1: "bank_len"},
        }
    else:
        print("> Static export (no dynamic_axes): shapes frozen to the dummy "
              "input sizes. Use --dynamic for variable H/W/bank_len.")

    try:
        torch.onnx.export(
            model, dummy, args.out,
            input_names=input_names, output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dyn,
        )
        print(f"> Saved ONNX -> {args.out}")

        if args.simplify:
            _simplify_onnx(args.out)
    except Exception as e:
        print(f"\n> Export failed: {e}\n> Dumping traced graph to localize the "
              f"offending op (look for the scope of aten::copy / aten::copy_) ...\n")
        traced = torch.jit.trace(model, dummy, check_trace=False)
        graph_str = str(traced.inlined_graph)
        for line in graph_str.splitlines():
            if "copy" in line or "index_put" in line or "slice" in line:
                print(line.strip())
        raise


if __name__ == "__main__":
    main()
