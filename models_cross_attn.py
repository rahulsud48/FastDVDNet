"""
Definition of the FastDVDnet model
Depthwise-separable version:
each Conv2d(3x3) is replaced by:
    depthwise 3x3 + pointwise 1x1
No no_orthog changes introduced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class ChannelToSpace(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        N, C, H, W = x.shape
        num_positions = self.scale * self.scale

        assert C % num_positions == 0, \
            f"Channels {C} must be divisible by scale^2={num_positions}"

        g = C // num_positions  # inferred dynamically: 256//4 = 64

        chunks = x.view(N, num_positions, g, H, W)
        chunks = chunks.view(N, self.scale, self.scale, g, H, W)
        chunks = chunks.permute(0, 3, 4, 1, 5, 2)
        out = chunks.contiguous().view(N, g, H * self.scale, W * self.scale)
        return out

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

class CreateKV(nn.Module):
    def __init__(self, ch, d=64):
        super().__init__()
        self.d = d
        self.grid_size = int(d**0.5)
        self.pool = nn.AdaptiveAvgPool2d(self.grid_size)
        
        # Pointwise convs are often faster on NPUs than Linear layers for 4D tensors
        self.to_k = nn.Conv2d(ch, ch, kernel_size=1)
        self.to_v = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        # x: (N, C, H, W)
        x_pooled = self.pool(x) # (N, C, grid, grid)
        
        k = self.to_k(x_pooled).flatten(2) # (N, C, d)
        v = self.to_v(x_pooled).flatten(2) # (N, C, d)
        return k, v

class CrossAttentionSkip(nn.Module):
    def __init__(self, ch, d=64, num_heads=4):
        super().__init__()
        self.ch = ch
        self.d = d
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        
        self.q_proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.out_proj = nn.Conv2d(ch, ch, kernel_size=1)
        
        # Learned positional bias for the d tokens
        # Helps the model remember which part of the 8x8 grid is which
        self.pos_emb = nn.Parameter(torch.randn(1, 1, d))
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, up_feat, k, v):
        N, C, H, W = up_feat.shape
        
        # 1. Project Query and prepare for Multi-Head
        # (N, heads, HW, head_dim)
        q = self.q_proj(up_feat).flatten(2).view(N, self.num_heads, self.head_dim, -1).transpose(-1, -2)
        
        # 2. Add positional info to Keys and prepare K/V
        # (N, heads, head_dim, d)
        k = (k + self.pos_emb).view(N, self.num_heads, self.head_dim, self.d)
        v = v.view(N, self.num_heads, self.head_dim, self.d)
        
        # 3. Scaled Dot-Product Attention
        # (N, heads, HW, d)
        attn = torch.matmul(q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # 4. Pull values and Reconstruct
        # (N, heads, HW, head_dim)
        out = torch.matmul(attn, v.transpose(-1, -2))
        
        # 5. Merge heads and Reshape
        out = out.transpose(-1, -2).contiguous().view(N, C, H, W)
        out = self.out_proj(out)
        
        return up_feat + self.gamma * out

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
            # nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            DSConv(num_in_frames * self.interm_ch, out_ch),
            # nn.BatchNorm2d(out_ch),
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
            # nn.BatchNorm2d(out_ch),
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
            # ChannelToSpace(scale=2)
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
            # nn.BatchNorm2d(in_ch), 
            nn.ReLU(inplace=True),
            DSConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """
    Optimized Denoising Block for FastDVDnet.
    Uses Cross-Attention to pull features from compressed skip connections.
    """
    def __init__(self, num_input_frames=3):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128
        latent_tokens = 64 # The 'd' dimension

        # Standard FastDVDnet layers (using Depthwise-Separable Convs)
        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

        # The Compression and Attention modules
        self.getKV0 = CreateKV(ch=self.chs_lyr0, d=latent_tokens)
        self.getKV1 = CreateKV(ch=self.chs_lyr1, d=latent_tokens)

        self.cross_attn1 = CrossAttentionSkip(ch=self.chs_lyr1, d=latent_tokens)
        self.cross_attn0 = CrossAttentionSkip(ch=self.chs_lyr0, d=latent_tokens)

    def forward(self, in0, in1, in2, noise_map):
        # 1. Input Layer
        # Concatenate frames and noise: (N, 12, H, W)
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        
        # --- COMPRESSION STEP 0 ---
        # Generate small KV tensors and immediately "release" x0 conceptually
        k0, v0 = self.getKV0(x0) 
        
        # 2. Downscale 0
        x1 = self.downc0(x0)
        
        # --- COMPRESSION STEP 1 ---
        k1, v1 = self.getKV1(x1)
        
        # 3. Downscale 1 (Bottleneck)
        x2 = self.downc1(x1)

        # 4. Upscale 2 (Moving back up)
        # x2 goes from chs_lyr2 -> chs_lyr1
        x2 = self.upc2(x2)
        
        # 5. CROSS-ATTENTION 1 
        # Instead of x1 + x2, we use x2 to query the compressed k1, v1
        x1_refined = self.cross_attn1(x2, k1, v1)
        x1_up = self.upc1(x1_refined)

        # 6. CROSS-ATTENTION 0
        # Final refinement at the highest resolution
        x0_refined = self.cross_attn0(x1_up, k0, v0)
        
        # 7. Output
        res = self.outc(x0_refined)

        # Standard residual denoising: clean = noisy - noise_estimate
        return in1 - res



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
    from torchinfo import summary
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastDVDnet().to(device)
    model.eval()

    input_data = torch.randn(1, 9, 1080, 1920).to(device)
    noise_map = torch.randn(1, 1, 1080, 1920).to(device)

    start_time = time.time()

    with torch.no_grad():
        output_data = model(input_data, noise_map)

    torch.cuda.synchronize()

    end_time = time.time()

    print("Using device:", device)
    print("Time taken:", end_time - start_time)
    print("Output shape:", output_data.shape)

    summary(
        model,
        input_data=(torch.randn(1, 9, 96, 96), torch.randn(1, 1, 96, 96)),
        col_names=["input_size", "output_size", "num_params"],
        depth=5
    )