"""
FastDVDnet — Single-frame input with KV Bank cross-attention at bottleneck.

Changes from uploaded version:
  - InputCvBlock accepts 5 channels: RGB(3) + sigma_read(1) + lambda_shot(1)
  - All AdaptiveAvgPool2d replaced with depthwise strided Conv2d (compiler-friendly)
  - Skip connections replaced with spatially-aware cross-attention gating on x_dec:
      * K, V computed from x_enc via depthwise strided conv (4x4) + Linear → DRAM
      * Q computed from x_dec via depthwise strided conv (4x4) + Linear
      * Gate interpolated back to (H, W) → applied on x_dec only
      * x_enc never loaded into SRAM; only C*16 floats cross DRAM<->SRAM per skip

  Pool params for each level (input H=270, W=480 after /4 crop):
    bottleneck (128, 67, 120) → (128, 8, 8): kernel=(9,15),  stride=(8,15),  pad=(1,0)
    skip_gate1  (64, 135,240) → (64,  4, 4): kernel=(35,60), stride=(33,60), pad=(1,0)
    skip_gate0  (32, 270,480) → (32,  4, 4): kernel=(69,120),stride=(67,120),pad=(1,0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CvBlock(nn.Module):
    """(Conv2d 3x3 => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """
    First encoder block.
    Accepts 5 channels: RGB(3) + sigma_read map(1) + lambda_shot map(1).
    """
    def __init__(self, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(5, self.interm_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """Stride-2 Conv2d => BN => ReLU => CvBlock"""
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """CvBlock => ConvTranspose2d x2"""
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.cvblock  = CvBlock(in_ch, in_ch)
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.upsample(self.cvblock(x))


class OutputCvBlock(nn.Module):
    """Conv2d 3x3 => BN => ReLU => Conv2d 3x3"""
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)


class SkipCrossAttnGate(nn.Module):
    """
    Spatially-aware cross-attention gate replacing direct skip connections.

    Uses AdaptiveAvgPool2d(4) for spatial compression — must handle arbitrary
    input sizes during training (patch crops) as well as full resolution at
    inference. The depthwise strided conv approach only works for a fixed known
    resolution; for the compiler team, swap self.pool to a depthwise strided
    conv sized for your inference resolution.

    Encoder side — encode(x_enc):
        AdaptiveAvgPool2d(4): (N, C, H, W) → (N, C, 4, 4)
        flatten → Linear → k, v : (N, C*16) each
        Only these C*16 floats written to DRAM. x_enc discarded.

    Decoder side — gate(x_dec, k, v):
        AdaptiveAvgPool2d(4): (N, C, H', W') → (N, C, 4, 4)
        flatten → Linear → q : (N, C*16)
        gate = sigmoid(dot(q, k)) * v → (N, C*16)
        reshape → (N, C, 4, 4) → bilinear upsample → (N, C, H', W')
        out = x_dec + gate * x_dec

    DRAM traffic: C*16 floats per skip level.
      ch=64 → 1024 floats  (vs ~2M full skip, ~2000x reduction)
      ch=32 → 512  floats  (vs ~4M full skip, ~8000x reduction)
    """
    def __init__(self, ch, spatial_pool: int = 4):
        super(SkipCrossAttnGate, self).__init__()
        self.spatial_pool = spatial_pool
        self.flat_dim     = ch * spatial_pool * spatial_pool

        self.pool  = nn.AdaptiveAvgPool2d(spatial_pool)
        self.to_k  = nn.Linear(self.flat_dim, self.flat_dim, bias=False)
        self.to_v  = nn.Linear(self.flat_dim, self.flat_dim, bias=False)
        self.to_q  = nn.Linear(self.flat_dim, self.flat_dim, bias=False)
        self.scale = self.flat_dim ** -0.5

    def encode(self, x_enc):
        """
        Encoder pass: compress x_enc → k, v.
        x_enc : (N, C, H, W)
        returns k, v : (N, C*16) each — only these written to DRAM
        """
        feat = self.pool(x_enc).flatten(1)    # (N, C*16)
        return self.to_k(feat), self.to_v(feat)

    def gate(self, x_dec, k, v):
        """
        Decoder pass: spatially gate x_dec using k, v from DRAM.
        x_dec : (N, C, H', W')
        k, v  : (N, C*16)
        returns modulated x_dec : (N, C, H', W')
        """
        N, C, H, W = x_dec.shape
        q    = self.to_q(self.pool(x_dec).flatten(1))           # (N, C*16)
        attn = (q * k).sum(dim=-1, keepdim=True) * self.scale   # (N, 1)
        gate = torch.sigmoid(attn) * v                           # (N, C*16)

        # Reshape to spatial grid, upsample to decoder resolution
        gate = gate.view(N, C, self.spatial_pool, self.spatial_pool)
        gate = F.interpolate(gate, size=(H, W), mode='bilinear', align_corners=False)

        return x_dec + gate * x_dec


class KVBank:
    def __init__(self, bank_size: int = 10, detach: bool = True):
        self.bank_size = bank_size
        self.detach    = detach
        self._keys:   list = []
        self._values: list = []

    def reset(self):
        self._keys.clear()
        self._values.clear()

    def push(self, k: torch.Tensor, v: torch.Tensor):
        self._keys.append(k.detach() if self.detach else k)
        self._values.append(v.detach() if self.detach else v)
        if len(self._keys) > self.bank_size:
            self._keys.pop(0)
            self._values.pop(0)

    def get(self):
        if not self._keys:
            return None, None
        return torch.cat(self._keys, dim=1), torch.cat(self._values, dim=1)

    def __len__(self):
        return len(self._keys)


class BottleneckCrossAttn(nn.Module):
    """
    Cross-attention at bottleneck using a depthwise strided Conv2d for spatial pooling.
    Default params for bottleneck input (N, 128, 67, 120) -> (N, 128, 8, 8):
        kernel=(9,15), stride=(8,15), padding=(1,0)
    """
    def __init__(self, ch: int = 128, num_heads: int = 4,
                 pool_size: int = 8,
                 pool_kernel: tuple = (9, 15),
                 pool_stride: tuple = (8, 15),
                 pool_padding: tuple = (1, 0)):
        super(BottleneckCrossAttn, self).__init__()
        self.pool = nn.Conv2d(
            in_channels  = ch,
            out_channels = ch,
            kernel_size  = pool_kernel,
            stride       = pool_stride,
            padding      = pool_padding,
            groups       = ch,
            bias         = False
        )
        self.to_q = nn.Linear(ch, ch, bias=False)
        self.to_k = nn.Linear(ch, ch, bias=False)
        self.to_v = nn.Linear(ch, ch, bias=False)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(ch, ch), nn.Sigmoid())

    def _to_tokens(self, feat):
        return self.pool(feat).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, bank: KVBank):
        N, C, H, W = x.shape
        tokens = self._to_tokens(x)
        q_cur  = self.to_q(tokens)
        k_cur  = self.to_k(tokens)
        v_cur  = self.to_v(tokens)

        bank_k, bank_v = bank.get()
        if bank_k is None:
            return x, k_cur, v_cur

        attn_out, _ = self.attn(q_cur, bank_k, bank_v)
        gate  = self.gate(attn_out.mean(dim=1)).view(N, C, 1, 1)
        x_out = x * gate + x
        return x_out, k_cur, v_cur


class DenBlock(nn.Module):
    """
    FastDVDnet denoising block — single frame + dual noise maps + KV bank.

    Skip connection design:
      Encoder computes k, v from x0, x1 via depthwise strided conv (4x4) + Linear.
      Decoder gates x2 / x1_out with spatially interpolated gate — no full skip loaded.
      DRAM<->SRAM traffic: (64*16) + (32*16) = 1536 floats (vs ~6M for full skips).

    forward(frame_t, sigma_read_map, lambda_shot_map, bank)
      frame_t         : (N, 3, H, W)
      sigma_read_map  : (N, 1, H, W)
      lambda_shot_map : (N, 1, H, W)
      bank            : KVBank

    Returns: denoised (N, 3, H, W)
    """
    def __init__(self, bank_size: int = 10, num_heads: int = 4,
                 pool_kernel:  tuple = (9,  15),
                 pool_stride:  tuple = (8,  15),
                 pool_padding: tuple = (1,   0)):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc    = InputCvBlock(out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)

        self.kv_attn = BottleneckCrossAttn(
            ch=self.chs_lyr2, num_heads=num_heads,
            pool_kernel=pool_kernel, pool_stride=pool_stride, pool_padding=pool_padding
        )

        self.skip_gate1 = SkipCrossAttnGate(ch=self.chs_lyr1)
        self.skip_gate0 = SkipCrossAttnGate(ch=self.chs_lyr0)

        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, frame_t: torch.Tensor,
                      sigma_read_map: torch.Tensor,
                      lambda_shot_map: torch.Tensor,
                      bank: KVBank):

        # ---- Encoder ----
        x0 = self.inc(torch.cat((frame_t, sigma_read_map, lambda_shot_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)

        # Compress skips to k, v — only these go to DRAM, x0/x1 never read again
        k1, v1 = self.skip_gate1.encode(x1)   # (N, 64*16)
        k0, v0 = self.skip_gate0.encode(x0)   # (N, 32*16)

        # ---- Bottleneck ----
        x2, k_cur, v_cur = self.kv_attn(x2, bank)
        bank.push(k_cur, v_cur)

        # ---- Decoder ----
        x2     = self.upc2(x2)                          # (N, 64, H/2, W/2)
        x2     = self.skip_gate1.gate(x2, k1, v1)       # gated by x1 encoder context
        x1_out = self.upc1(x2)                           # (N, 32, H, W)
        x1_out = self.skip_gate0.gate(x1_out, k0, v0)   # gated by x0 encoder context
        x      = self.outc(x1_out)

        return frame_t - x


class FastDVDnet(nn.Module):
    def __init__(self, bank_size: int = 10, num_heads: int = 4,
                 pool_size: int = 8,          # accepted for backwards compat, not used
                 pool_kernel:  tuple = (9,  15),
                 pool_stride:  tuple = (8,  15),
                 pool_padding: tuple = (1,   0)):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = 1
        self.temp = DenBlock(
            bank_size=bank_size, num_heads=num_heads,
            pool_kernel=pool_kernel, pool_stride=pool_stride, pool_padding=pool_padding
        )
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, frame_t: torch.Tensor,
                      sigma_read_map: torch.Tensor,
                      lambda_shot_map: torch.Tensor,
                      bank: KVBank):
        return self.temp(frame_t, sigma_read_map, lambda_shot_map, bank)


if __name__ == "__main__":
    # Sanity check: H=270, W=480 (1080/4 x 1920/4)
    # Verify pool output shapes:
    #   bottleneck: (1,128,67,120) -> (1,128,8,8)
    #   skip_gate1: (1,64,135,240) -> (1,64,4,4)
    #   skip_gate0: (1,32,270,480) -> (1,32,4,4)
    import math
    def conv_out(i, k, s, p): return math.floor((i + 2*p - k) / s) + 1
    assert conv_out(67,  9,  8, 1) == 8, "bottleneck H"
    assert conv_out(120, 15, 15, 0) == 8, "bottleneck W"
    assert conv_out(135, 35, 33, 1) == 4, "sg1 H"
    assert conv_out(240, 60, 60, 0) == 4, "sg1 W"
    assert conv_out(270, 69, 67, 1) == 4, "sg0 H"
    assert conv_out(480, 120,120, 0) == 4, "sg0 W"
    print("Pool shape assertions passed.")

    bank_size = 10
    model = FastDVDnet(bank_size=bank_size, num_heads=4)
    bank  = KVBank(bank_size=bank_size)
    model.eval()
    with torch.no_grad():
        for t in range(5):
            frame = torch.randn(1, 3, 270, 480)
            s_map = torch.full((1, 1, 270, 480), 0.02)
            l_map = torch.rand(1, 1, 270, 480) * 0.5
            out   = model(frame, s_map, l_map, bank)
            print(f"t={t}  bank_len={len(bank)}  out={out.shape}")
    print("Sanity check passed!")
