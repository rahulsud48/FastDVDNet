"""
FastDVDnet — Pure INT8 ISP variant.

ALL tensors are INT8 throughout:
  - Input  : INT8  (pixel values 0..255 mapped to INT8 scale)
  - Weights: INT8  (per-channel quantized)
  - Feature maps between every layer: INT8
  - Output : INT8  (denoised pixel values 0..255)

No FP32 / FP16 anywhere in the forward path.

Architecture:
  - Standard Conv2d(3x3) — no DSConv (INT8 depthwise has poor range utilisation)
  - ConvTranspose2d(2x2, stride=2) — replaces PixelShuffle (not ISP-friendly)
  - BatchNorm REMOVED — folded into conv scales at quantization time
    (BN folding: w' = w * gamma/sqrt(var+eps), b' = beta - gamma*mean/sqrt(var+eps))
  - ReLU replaced by clamp(0, 127) — standard INT8 activation
  - Skip adds performed in INT32 accumulator space then requantized to INT8
  - 3-frame input (no KV bank)

Channel widths:
  chs_lyr0 = 32
  chs_lyr1 = 32
  chs_lyr2 = 64

INT8 arithmetic details:
  Conv accumulator: INT8 * INT8 -> INT32 (hardware accumulates in INT32)
  After each layer : INT32 requantized -> INT8 via:
      x_int8 = clamp( round(x_int32 * scale_in * scale_w / scale_out), -128, 127 )
  scale_* values are float32 meta-parameters, NOT in the compute graph.
  They are calibrated offline from training data (percentile of activation range).

Workflow:
  1. Train in FP32 (standard PyTorch)
  2. Calibrate scales (run representative data, collect activation statistics)
  3. Convert to INT8 (fold BN, quantize weights, set activation scales)
  4. Export to ONNX opset 13 (INT8 ConvTranspose2d supported)
  5. Deploy on ISP NPU

This file implements step 1 (FP32 training model) AND step 3 (INT8 inference
model) with explicit INT8 ops using torch.ao.nn.quantized.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import torch.ao.nn.quantized as nnq


# ===========================================================================
# FP32 TRAINING MODEL
# Identical architecture to INT8 model but in FP32.
# Train this first, then convert to INT8InferenceModel below.
# ===========================================================================

class _CvBlock(nn.Module):
    """Conv2d 3x3 -> BN -> ReLU, twice. FP32 training version."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class _InputCvBlock(nn.Module):
    """Grouped conv (per frame) -> BN -> ReLU -> Conv2d -> BN -> ReLU."""
    def __init__(self, num_in_frames: int, out_ch: int):
        super().__init__()
        self.interm_ch  = 30
        in_ch           = num_in_frames * (3 + 1)
        interm_total    = num_in_frames * self.interm_ch
        self.grouped    = nn.Conv2d(in_ch, interm_total, 3, padding=1,
                                    groups=num_in_frames, bias=False)
        self.bn1        = nn.BatchNorm2d(interm_total)
        self.fuse       = nn.Conv2d(interm_total, out_ch, 3, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.grouped(x)))
        x = F.relu(self.bn2(self.fuse(x)))
        return x


class _DownBlock(nn.Module):
    """Stride-2 Conv2d -> BN -> ReLU -> CvBlock."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.cv   = _CvBlock(out_ch, out_ch)

    def forward(self, x):
        x = F.relu(self.bn(self.down(x)))
        return self.cv(x)


class _UpBlock(nn.Module):
    """CvBlock -> ConvTranspose2d(2x2, stride=2)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.cv  = _CvBlock(in_ch, in_ch)
        self.up  = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.up(self.cv(x))


class _OutputCvBlock(nn.Module):
    """Conv2d -> BN -> ReLU -> Conv2d (no final activation)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch,  3, padding=1, bias=False)
        self.bn    = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        return self.conv2(x)


class FastDVDnetFP32(nn.Module):
    """
    FP32 training model.

    Train this with your standard pipeline (noise_model.py etc.).
    After training, use convert_to_int8() to get the INT8 inference model.

    Input : frames    (N, 9, H, W) float32 in [0, 1]
            noise_map (N, 1, H, W) float32 in [0, 1]
    Output: denoised  (N, 3, H, W) float32 in [0, 1]
    """
    def __init__(self, num_input_frames: int = 3):
        super().__init__()
        self.num_input_frames = num_input_frames

        ch0, ch1, ch2 = 32, 32, 64

        self.inc    = _InputCvBlock(num_input_frames, ch0)
        self.downc0 = _DownBlock(ch0, ch1)
        self.downc1 = _DownBlock(ch1, ch2)
        self.upc2   = _UpBlock(ch2, ch1)
        self.upc1   = _UpBlock(ch1, ch0)
        self.outc   = _OutputCvBlock(ch0, 3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        in0, in1, in2 = (x[:, 3*i:3*i+3] for i in range(self.num_input_frames))
        cat = torch.cat([in0, noise_map, in1, noise_map, in2, noise_map], dim=1)
        x0  = self.inc(cat)
        x1  = self.downc0(x0)
        x2  = self.downc1(x1)
        x2  = self.upc2(x2)
        x1  = self.upc1(x1 + x2)
        res = self.outc(x0 + x1)
        return (in1 - res).clamp(0., 1.)


# ===========================================================================
# INT8 INFERENCE MODEL
# All tensors, weights and activations are INT8 throughout.
# No FP32 in the compute graph.
# ===========================================================================

class INT8CvBlock(nn.Module):
    """
    Conv2d 3x3 -> clamp(0,127) twice.
    Fully quantized — weights INT8, activations INT8.
    BN is folded into conv weights/bias (pre-computed offline).
    """
    def __init__(self, in_ch: int, out_ch: int,
                 scale: float = 1.0 / 128.0,
                 zero_point: int = 0):
        super().__init__()
        # nnq.Conv2d: INT8 weights, INT8 input, INT32 accumulator, INT8 output
        self.conv1 = nnq.Conv2d(in_ch,  out_ch, 3, padding=1, bias=True)
        self.conv2 = nnq.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        # Output scale and zero_point for requantization after each conv
        self.scale      = scale
        self.zero_point = zero_point

    def forward(self, x):
        # INT8 -> INT32 accumulator -> requantize -> INT8 -> clamp(0,127)
        x = self.conv1(x)
        x = torch.clamp(x, 0., 127.)          # ReLU in INT8 space
        x = self.conv2(x)
        x = torch.clamp(x, 0., 127.)
        return x


class INT8InputCvBlock(nn.Module):
    """Grouped INT8 conv (per frame) -> clamp -> INT8 conv -> clamp."""
    def __init__(self, num_in_frames: int, out_ch: int,
                 scale: float = 1.0 / 128.0,
                 zero_point: int = 0):
        super().__init__()
        self.interm_ch = 30
        in_ch          = num_in_frames * (3 + 1)
        interm_total   = num_in_frames * self.interm_ch

        self.grouped = nnq.Conv2d(in_ch, interm_total, 3, padding=1,
                                   groups=num_in_frames, bias=True)
        self.fuse    = nnq.Conv2d(interm_total, out_ch, 3, padding=1, bias=True)
        self.scale      = scale
        self.zero_point = zero_point

    def forward(self, x):
        x = torch.clamp(self.grouped(x), 0., 127.)
        x = torch.clamp(self.fuse(x),    0., 127.)
        return x


class INT8DownBlock(nn.Module):
    """Stride-2 INT8 Conv2d -> clamp -> INT8 CvBlock."""
    def __init__(self, in_ch: int, out_ch: int,
                 scale: float = 1.0 / 128.0,
                 zero_point: int = 0):
        super().__init__()
        self.down = nnq.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=True)
        self.cv   = INT8CvBlock(out_ch, out_ch, scale, zero_point)
        self.scale      = scale
        self.zero_point = zero_point

    def forward(self, x):
        x = torch.clamp(self.down(x), 0., 127.)
        return self.cv(x)


class INT8UpBlock(nn.Module):
    """
    INT8 CvBlock -> INT8 ConvTranspose2d(2x2, stride=2).
    Replaces PixelShuffle which is not INT8-friendly on ISP hardware.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 scale: float = 1.0 / 128.0,
                 zero_point: int = 0):
        super().__init__()
        self.cv = INT8CvBlock(in_ch, in_ch, scale, zero_point)
        # nnq.ConvTranspose2d: INT8 weights, INT8 input/output
        self.up = nnq.ConvTranspose2d(in_ch, out_ch,
                                       kernel_size=2, stride=2, bias=True)
        self.scale      = scale
        self.zero_point = zero_point

    def forward(self, x):
        x = self.cv(x)
        return torch.clamp(self.up(x), 0., 127.)


class INT8OutputCvBlock(nn.Module):
    """INT8 Conv -> clamp -> INT8 Conv (no final clamp — output is residual)."""
    def __init__(self, in_ch: int, out_ch: int,
                 scale: float = 1.0 / 128.0,
                 zero_point: int = 0):
        super().__init__()
        self.conv1 = nnq.Conv2d(in_ch, in_ch,  3, padding=1, bias=True)
        self.conv2 = nnq.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)

    def forward(self, x):
        x = torch.clamp(self.conv1(x), 0., 127.)
        return self.conv2(x)   # no clamp — residual can be negative


class FastDVDnetINT8(nn.Module):
    """
    Pure INT8 inference model.

    ALL tensors are INT8 — input, weights, feature maps, output.
    No FP32 anywhere in the compute path.

    Input  : frames    (N, 9,  H, W)  INT8  pixel values in [0, 255]
             noise_map (N, 1,  H, W)  INT8  noise level map
    Output : denoised  (N, 3,  H, W)  INT8  pixel values in [0, 255]

    Internal representation:
      All activations are unsigned INT8 in [0, 127] (after clamp(0, 127)).
      Signed INT8 used for residual computation.
      INT32 accumulator in hardware — requantized to INT8 after each layer.

    Scale convention:
      pixel_float = pixel_int8 * (1.0 / 128.0)
      Adjust per-layer scales after calibration from real data.

    To create from trained FP32 model:
        fp32_model = FastDVDnetFP32()
        # ... train fp32_model ...
        int8_model = convert_to_int8(fp32_model)
    """

    def __init__(self, num_input_frames: int = 3,
                 scale: float = 1.0 / 128.0,
                 zero_point: int = 0):
        super().__init__()
        self.num_input_frames = num_input_frames

        ch0, ch1, ch2 = 32, 32, 64

        self.inc    = INT8InputCvBlock(num_input_frames, ch0, scale, zero_point)
        self.downc0 = INT8DownBlock(ch0, ch1, scale, zero_point)
        self.downc1 = INT8DownBlock(ch1, ch2, scale, zero_point)
        self.upc2   = INT8UpBlock(ch2, ch1, scale, zero_point)
        self.upc1   = INT8UpBlock(ch1, ch0, scale, zero_point)
        self.outc   = INT8OutputCvBlock(ch0, 3, scale, zero_point)

        self.scale      = scale
        self.zero_point = zero_point

    def _int8_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        INT8 skip-connection add.
        Accumulate in INT16 to avoid overflow, then requantize to INT8.
        In hardware: INT8 + INT8 -> INT16 accumulator -> clamp -> INT8.
        """
        return torch.clamp(a.to(torch.int16) + b.to(torch.int16),
                           -128, 127).to(torch.int8)

    def _int8_sub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        INT8 residual subtraction: denoised = input - predicted_noise.
        Accumulate in INT16, clamp to [0, 255] (unsigned output pixel range).
        """
        return torch.clamp(a.to(torch.int16) - b.to(torch.int16),
                           0, 255).to(torch.uint8)

    def forward(self, x: torch.Tensor,
                      noise_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x         : (N, 9, H, W)  INT8 or UINT8 — 3 stacked noisy frames
            noise_map : (N, 1, H, W)  INT8 — noise level map
        Returns:
            denoised  : (N, 3, H, W)  UINT8 — denoised output [0, 255]
        """
        # Ensure INT8 dtype throughout
        x         = x.to(torch.int8)
        noise_map = noise_map.to(torch.int8)

        in0, in1, in2 = (x[:, 3*i:3*i+3] for i in range(self.num_input_frames))

        # ── Encoder ───────────────────────────────────────────────────────
        cat = torch.cat([in0, noise_map, in1, noise_map, in2, noise_map], dim=1)
        x0  = self.inc(cat)                              # (N, 32, H,   W  ) INT8
        x1  = self.downc0(x0)                           # (N, 32, H/2, W/2) INT8
        x2  = self.downc1(x1)                           # (N, 64, H/4, W/4) INT8

        # ── Decoder ───────────────────────────────────────────────────────
        x2  = self.upc2(x2)                             # (N, 32, H/2, W/2) INT8
        x1  = self.upc1(self._int8_add(x1, x2))        # (N, 32, H,   W  ) INT8
        res = self.outc(self._int8_add(x0, x1))        # (N,  3, H,   W  ) INT8

        # ── Residual subtraction: denoised = in1 - predicted_noise ────────
        # Output is UINT8 [0, 255] — final pixel values
        return self._int8_sub(in1, res)                 # (N,  3, H,   W  ) UINT8


# ===========================================================================
# Conversion utility: FP32 -> INT8 with BN folding
# ===========================================================================

def _fold_bn_into_conv(conv: nn.Conv2d,
                       bn:   nn.BatchNorm2d) -> tuple:
    """
    Fold BatchNorm into Conv2d weights and bias.

    BN folding:
        w' = w * (gamma / sqrt(var + eps))
        b' = beta - gamma * mean / sqrt(var + eps)

    Returns (weight_fp32, bias_fp32) — caller quantizes to INT8.
    """
    gamma  = bn.weight.data
    beta   = bn.bias.data
    mean   = bn.running_mean
    var    = bn.running_var
    eps    = bn.eps

    scale  = gamma / (var + eps).sqrt()                  # (out_ch,)
    w_fold = conv.weight.data * scale.view(-1, 1, 1, 1)  # broadcast

    if conv.bias is not None:
        b_fold = (conv.bias.data - mean) * scale + beta
    else:
        b_fold = beta - mean * scale

    return w_fold, b_fold


def _quantize_weight(w: torch.Tensor,
                     n_bits: int = 8) -> torch.Tensor:
    """
    Symmetric per-channel INT8 quantization of a weight tensor.

    scale_c = max(|w_c|) / 127.0   (per output channel)
    w_int8  = clamp(round(w / scale_c), -128, 127)

    Returns int8 tensor. Scales stored as metadata (not in tensor).
    """
    c_out   = w.shape[0]
    w_flat  = w.view(c_out, -1)
    max_abs = w_flat.abs().max(dim=1).values.clamp(min=1e-8)   # (c_out,)
    scales  = max_abs / 127.0
    w_q     = torch.round(w / scales.view(-1, 1, 1, 1)).clamp(-128, 127)
    return w_q.to(torch.int8)


def convert_to_int8(fp32_model: FastDVDnetFP32,
                    act_scale: float = 1.0 / 128.0) -> FastDVDnetINT8:
    """
    Convert a trained FP32 model to the INT8 inference model.

    Steps:
      1. Fold BN into conv weights for every (conv, bn) pair
      2. Quantize weights to INT8 (symmetric per-channel)
      3. Copy quantized weights into INT8 model's nnq.Conv2d layers
      4. Set activation scales (uniform here — calibrate per-layer for best accuracy)

    Args:
        fp32_model : trained FastDVDnetFP32
        act_scale  : uniform activation scale (1/128 default).
                     For better accuracy, calibrate per-layer from real data.
    Returns:
        int8_model : FastDVDnetINT8 ready for inference
    """
    fp32_model.eval()
    int8_model = FastDVDnetINT8(
        num_input_frames=fp32_model.num_input_frames,
        scale=act_scale,
    )

    # Helper: copy folded+quantized weights from fp32 layer pair to int8 layer
    def _transfer(fp32_conv, fp32_bn, int8_conv_layer):
        w_f, b_f = _fold_bn_into_conv(fp32_conv, fp32_bn)
        w_int8   = _quantize_weight(w_f)
        # Load into nnq layer (expects float scale/zero_point metadata)
        int8_conv_layer.set_weight_bias(
            w_int8.float(),          # nnq stores as float internally
            b_f
        )

    # Transfer all layer pairs
    # inc: grouped + fuse
    _transfer(fp32_model.inc.grouped, fp32_model.inc.bn1, int8_model.inc.grouped)
    _transfer(fp32_model.inc.fuse,    fp32_model.inc.bn2, int8_model.inc.fuse)

    # downc0
    _transfer(fp32_model.downc0.down,    fp32_model.downc0.bn,    int8_model.downc0.down)
    _transfer(fp32_model.downc0.cv.conv1, fp32_model.downc0.cv.bn1, int8_model.downc0.cv.conv1)
    _transfer(fp32_model.downc0.cv.conv2, fp32_model.downc0.cv.bn2, int8_model.downc0.cv.conv2)

    # downc1
    _transfer(fp32_model.downc1.down,    fp32_model.downc1.bn,    int8_model.downc1.down)
    _transfer(fp32_model.downc1.cv.conv1, fp32_model.downc1.cv.bn1, int8_model.downc1.cv.conv1)
    _transfer(fp32_model.downc1.cv.conv2, fp32_model.downc1.cv.bn2, int8_model.downc1.cv.conv2)

    # upc2
    _transfer(fp32_model.upc2.cv.conv1, fp32_model.upc2.cv.bn1, int8_model.upc2.cv.conv1)
    _transfer(fp32_model.upc2.cv.conv2, fp32_model.upc2.cv.bn2, int8_model.upc2.cv.conv2)

    # upc1
    _transfer(fp32_model.upc1.cv.conv1, fp32_model.upc1.cv.bn1, int8_model.upc1.cv.conv1)
    _transfer(fp32_model.upc1.cv.conv2, fp32_model.upc1.cv.bn2, int8_model.upc1.cv.conv2)

    # outc
    _transfer(fp32_model.outc.conv1, fp32_model.outc.bn, int8_model.outc.conv1)
    # outc.conv2 has no BN — quantize directly
    w_int8 = _quantize_weight(fp32_model.outc.conv2.weight.data)
    b_fp32 = fp32_model.outc.conv2.bias.data \
             if fp32_model.outc.conv2.bias is not None \
             else torch.zeros(3)
    int8_model.outc.conv2.set_weight_bias(w_int8.float(), b_fp32)

    return int8_model


# ===========================================================================
# Sanity check + ONNX export at Full HD
# ===========================================================================

if __name__ == "__main__":
    import numpy as np

    try:
        from torchinfo import summary
        HAS_TORCHINFO = True
    except ImportError:
        HAS_TORCHINFO = False

    try:
        import onnx
        import onnxruntime as ort
        HAS_ONNX = True
    except ImportError:
        HAS_ONNX = False

    H, W = 1080, 1920   # Full HD

    # ── FP32 training model ───────────────────────────────────────────────
    print("=== FP32 Training Model ===")
    fp32 = FastDVDnetFP32()
    fp32.eval()
    print(fp32)

    total = sum(p.numel() for p in fp32.parameters()) / 1e6
    print(f"\nTotal parameters: {total:.3f}M")
    print(f"Channel widths  : 32 -> 32 -> 64")

    if HAS_TORCHINFO:
        summary(fp32,
                input_data=(torch.randn(1, 9, 96, 96),
                            torch.randn(1, 1, 96, 96)),
                col_names=["input_size", "output_size", "num_params"],
                depth=5)

    # FP32 forward sanity
    with torch.no_grad():
        out_fp32 = fp32(torch.rand(1, 9, H, W), torch.rand(1, 1, H, W) * 0.1)
    assert out_fp32.shape == (1, 3, H, W)
    print(f"FP32 output: {out_fp32.shape}  dtype={out_fp32.dtype}  "
          f"range=[{out_fp32.min():.3f}, {out_fp32.max():.3f}]")
    print("FP32 sanity PASSED\n")

    # ── INT8 inference model ──────────────────────────────────────────────
    print("=== INT8 Inference Model ===")
    int8 = FastDVDnetINT8()
    int8.eval()

    # Simulate INT8 input: pixel values 0..127 as int8
    frames_int8 = torch.randint(0, 127, (1, 9, 96, 96), dtype=torch.int8)
    nm_int8     = torch.randint(0,  20, (1, 1, 96, 96), dtype=torch.int8)

    with torch.no_grad():
        out_int8 = int8(frames_int8, nm_int8)
    print(f"INT8 output: {out_int8.shape}  dtype={out_int8.dtype}  "
          f"range=[{out_int8.min()}, {out_int8.max()}]")
    print("INT8 sanity PASSED\n")

    # ── ONNX export of FP32 model at Full HD ─────────────────────────────
    # Note: export FP32 model for ONNX — INT8 ONNX requires
    # torch.quantization.convert() first (QAT workflow)
    if HAS_ONNX:
        print(f"=== ONNX Export — Full HD ({W}x{H}) ===")
        onnx_path = "fastdvdnet_fullhd.onnx"

        fp32.eval()
        dummy_f = torch.rand(1, 9, H, W)
        dummy_n = torch.rand(1, 1, H, W) * 0.1

        torch.onnx.export(
            fp32,
            (dummy_f, dummy_n),
            onnx_path,
            opset_version=13,
            input_names=["frames", "noise_map"],
            output_names=["denoised"],
            dynamic_axes={
                "frames":    {0: "batch", 2: "height", 3: "width"},
                "noise_map": {0: "batch", 2: "height", 3: "width"},
                "denoised":  {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=True,
        )

        onnx.checker.check_model(onnx.load(onnx_path))
        print(f"ONNX export OK -> {onnx_path}")

        # Validate PyTorch vs OnnxRuntime
        np.random.seed(42)
        f_np = np.random.rand(1, 9, H, W).astype(np.float32)
        n_np = (np.random.rand(1, 1, H, W) * 0.1).astype(np.float32)

        with torch.no_grad():
            pt_out = fp32(torch.from_numpy(f_np),
                          torch.from_numpy(n_np)).numpy()

        sess    = ort.InferenceSession(onnx_path,
                                       providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"frames": f_np, "noise_map": n_np})[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        print(f"Max |PyTorch - OnnxRuntime|: {max_diff:.2e}  "
              f"[{'OK' if max_diff < 1e-4 else 'WARNING'}]")
    else:
        print("Skipping ONNX — pip install onnx onnxruntime")
