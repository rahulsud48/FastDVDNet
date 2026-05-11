"""
Definition of the FastDVDnet model
Depthwise-separable version:
each Conv2d(3x3) is replaced by:
    depthwise 3x3 + pointwise 1x1
No no_orthog changes introduced.
"""

import torch
import torch.nn as nn



class DSConv(nn.Module):
    """Depthwise 3x3 + Pointwise 1x1"""
    def __init__(self, in_ch, out_ch, stride=1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False
            ),
            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, x):
        return self.conv(x)


class CvBlock(nn.Module):
    """(DSConv => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DSConv(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """(Grouped Conv with num_in_frames groups => BN => ReLU) + (DSConv => BN => ReLU)"""
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(
                num_in_frames * (3 + 1),
                num_in_frames * self.interm_ch,
                kernel_size=3,
                padding=1,
                groups=num_in_frames,
                bias=False
            ),
            nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            DSConv(num_in_frames * self.interm_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """Downscale + (DSConv => BN => ReLU)*2"""
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, out_ch, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """(DSConv => BN => ReLU)*2 + Upscale"""
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            DSConv(in_ch, out_ch * 4),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    """DSConv => BN => ReLU => DSConv"""
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            DSConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)

class ChannelAttentionGate(nn.Module):
    """SE-style channel attention on skip connection"""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # squeeze: (N, C, 1, 1)
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid()
        )

    def forward(self, skip):
        w = self.gate(skip).view(skip.shape[0], -1, 1, 1)
        return skip * w  # re-weighted channels

class CrossAttentionSkip(nn.Module):
    """
    Q from upblock output, KV from downblock skip.
    Operates on spatially pooled features to keep cost manageable.
    """
    def __init__(self, ch, num_heads=4, pool_size=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)  # compress spatial dims
        self.to_q = nn.Linear(ch, ch)
        self.to_kv = nn.Linear(ch, ch * 2)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(ch, ch), nn.Sigmoid())

    def forward(self, up_feat, skip_feat):
        N, C, H, W = skip_feat.shape
        # Pool and flatten spatial dims → tokens
        q_pooled = self.pool(up_feat).flatten(2).transpose(1, 2)   # (N, S, C)
        kv_pooled = self.pool(skip_feat).flatten(2).transpose(1, 2) # (N, S, C)

        q = self.to_q(q_pooled)
        k, v = self.to_kv(kv_pooled).chunk(2, dim=-1)

        attn_out, _ = self.attn(q, k, v)  # (N, S, C)
        
        # Use attention output as a channel gate on the full-res skip
        gate = self.gate(attn_out.mean(dim=1)).view(N, C, 1, 1)
        return skip_feat * gate  # modulate full-res skip features


class CBAM(nn.Module):
    def __init__(self, ch, reduction=8, kernel_size=7):
        super().__init__()
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attn(x).view(x.shape[0], -1, 1, 1)
        x = x * ca
        # Spatial attention (avg + max pool along channels)
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        sa = self.spatial_attn(torch.cat([avg_map, max_map], dim=1))
        return x * sa

class DenBlock(nn.Module):
    """Definition of the denoising block of FastDVDnet."""
    def __init__(self, num_input_frames=3):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

        # ── Option 1: Channel Attention Gate (SE-style) ─────────────────── ACTIVE
        # self.ca1 = ChannelAttentionGate(ch=self.chs_lyr1)
        # self.ca0 = ChannelAttentionGate(ch=self.chs_lyr0)

        # ── Option 2: Cross-Attention Skip ──────────────────────────────── INACTIVE
        # self.cross1 = CrossAttentionSkip(ch=self.chs_lyr1)
        # self.cross0 = CrossAttentionSkip(ch=self.chs_lyr0)

        # ── Option 3: CBAM ──────────────────────────────────────────────── INACTIVE
        self.cbam2 = CBAM(ch=self.chs_lyr2)   # applied at bottleneck (cheapest)
        self.cbam1 = CBAM(ch=self.chs_lyr1)
        self.cbam0 = CBAM(ch=self.chs_lyr0)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)

        x2 = self.upc2(x2)

        # ── Option 1: Channel Attention Gate ────────────────────────────── ACTIVE
        # Re-weights skip connection channels before adding upsampled features.
        # ca1 learns which channels of x1 are most informative given noise level.
        # x1 = self.upc1(self.ca1(x1) + x2)

        # ── Option 2: Cross-Attention Skip ──────────────────────────────── INACTIVE
        # Q from upc2 output (x2), KV from downblock skip (x1).
        # Lets upsampled features query which parts of the skip are relevant.
        # x1 = self.upc1(self.cross1(up_feat=x2, skip_feat=x1) + x2)

        # ── Option 3: CBAM ──────────────────────────────────────────────── INACTIVE
        # Apply spatial + channel attention on the bottleneck (x2) where maps
        # are smallest (H/4 x W/4), then plain residual add for skip connections.
        x2 = self.cbam2(x2)
        x1 = self.upc1(self.cbam1(x1) + x2)

        # ── shared output block (all options) ───────────────────────────────────
        # x = self.outc(self.ca0(x0) + x1)   # Option 1: attended x0 skip

        # Option 2 output:
        # x = self.outc(self.cross0(up_feat=x1, skip_feat=x0) + x1)

        # Option 3 output:
        x = self.outc(self.cbam0(x0) + x1)

        x = in1 - x
        return x


class FastDVDnet(nn.Module):
    def __init__(self, num_input_frames=3):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp = DenBlock(num_input_frames=3)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):
        x0, x1, x2 = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))
        x = self.temp(x0, x1, x2, noise_map)
        return x


if __name__ == "__main__":
    # import onnx
    # import onnxruntime as ort
    import numpy as np
    from torchinfo import summary
 
    model = FastDVDnet()
    print(model)
 
    summary(
        model,
        input_data=(torch.randn(1, 9, 96, 96), torch.randn(1, 1, 96, 96)),
        col_names=["input_size", "output_size", "num_params"],
        depth=5
    )
 
    # # --- Export to ONNX ---
    # model.eval()
    # dummy_frames    = torch.randn(1, 9, 96, 96)
    # dummy_noise_map = torch.randn(1, 1, 96, 96)
 
    # torch.onnx.export(
    #     model,
    #     (dummy_frames, dummy_noise_map),
    #     "fastdvdnet.onnx",
    #     opset_version=11,
    #     input_names=["frames", "noise_map"],
    #     output_names=["denoised"],
    #     do_constant_folding=True,
    # )
 
    # onnx.checker.check_model(onnx.load("fastdvdnet.onnx"))
    # print("ONNX export OK → fastdvdnet.onnx")
 
    # # --- Validate PyTorch vs OnnxRuntime ---
    # np.random.seed(42)
    # frames_np    = np.random.randn(1, 9, 96, 96).astype(np.float32)
    # noise_map_np = np.random.randn(1, 1, 96, 96).astype(np.float32)
 
    # with torch.no_grad():
    #     pt_out = model(torch.from_numpy(frames_np), torch.from_numpy(noise_map_np)).numpy()
 
    # sess    = ort.InferenceSession("fastdvdnet.onnx", providers=["CPUExecutionProvider"])
    # ort_out = sess.run(None, {"frames": frames_np, "noise_map": noise_map_np})[0]
 
    # max_diff = float(np.max(np.abs(pt_out - ort_out)))
    # print(f"Max |PyTorch − OnnxRuntime| diff: {max_diff:.2e}  "
    #       f"({'OK' if max_diff < 1e-4 else 'WARNING'})")