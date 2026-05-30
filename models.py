"""
FastDVDnet — Single-frame input with KV Bank cross-attention at bottleneck.

Changes from uploaded version:
  - InputCvBlock now accepts 5 channels: RGB(3) + sigma_read(1) + lambda_shot(1)
  - AdaptiveAvgPool2d replaced with depthwise strided Conv2d for compiler compatibility
    Input to pool: (N, 128, 67, 120) for H=270, W=480 (i.e. 1080/4 x 1920/4)
    kernel=(9,15), stride=(8,15), padding=(1,0), groups=128 → output (N, 128, 8, 8)
"""

import torch
import torch.nn as nn


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
        # 3 (RGB) + 1 (sigma_read) + 1 (lambda_shot) = 5 input channels
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

    Replaces AdaptiveAvgPool2d(pool_size) with a learnable depthwise conv that
    produces the same (N, ch, pool_size, pool_size) output shape.

    For the default input resolution 1080x1920 (stored as H=270, W=480 after /4 crop),
    the bottleneck feature map arriving here is (N, 128, 67, 120) after two stride-2
    downsamples. To reach (N, 128, 8, 8):

        H: kernel=9, stride=8, padding=1
           → floor((67 + 2*1 - 9) / 8) + 1 = floor(60/8) + 1 = 8  ✓
        W: kernel=15, stride=15, padding=0
           → floor((120 + 0 - 15) / 15) + 1 = floor(105/15) + 1 = 8  ✓

    If you change input resolution, recompute kernel/stride/padding accordingly and
    pass them as pool_kernel, pool_stride, pool_padding arguments.
    """
    def __init__(self, ch: int = 128, num_heads: int = 4,
                 pool_size: int = 8,
                 pool_kernel: tuple = (9, 15),
                 pool_stride: tuple = (8, 15),
                 pool_padding: tuple = (1, 0)):
        super(BottleneckCrossAttn, self).__init__()

        # Depthwise strided conv replaces AdaptiveAvgPool2d.
        # groups=ch → no channel mixing (same as pooling behaviour).
        # Learnable weights give the model freedom to learn better than avg pooling.
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
        # pool: (N, C, H, W) → (N, C, pool_size, pool_size)
        # flatten+transpose: → (N, pool_size*pool_size, C)
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

    forward(frame_t, sigma_read_map, lambda_shot_map, bank)
      frame_t         : (N, 3, H, W)  noisy RGB in [0,1]
      sigma_read_map  : (N, 1, H, W)  AWGN std  (flat scalar broadcast)
      lambda_shot_map : (N, 1, H, W)  Poisson lambda (spatially varying)
      bank            : KVBank

    Returns: denoised (N, 3, H, W)
    """
    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8,
                 pool_kernel: tuple = (9, 15),
                 pool_stride: tuple = (8, 15),
                 pool_padding: tuple = (1, 0)):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc    = InputCvBlock(out_ch=self.chs_lyr0)   # 5ch input
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.kv_attn = BottleneckCrossAttn(ch=self.chs_lyr2,
                                           num_heads=num_heads,
                                           pool_size=pool_size,
                                           pool_kernel=pool_kernel,
                                           pool_stride=pool_stride,
                                           pool_padding=pool_padding)
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
        # Concat: RGB + sigma_read + lambda_shot = 5 channels
        x0 = self.inc(torch.cat((frame_t, sigma_read_map, lambda_shot_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)

        x2, k_cur, v_cur = self.kv_attn(x2, bank)
        bank.push(k_cur, v_cur)

        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        x  = self.outc(x0 + x1)

        return frame_t - x


class FastDVDnet(nn.Module):
    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8,
                 pool_kernel: tuple = (9, 15),
                 pool_stride: tuple = (8, 15),
                 pool_padding: tuple = (1, 0)):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = 1
        self.temp = DenBlock(bank_size=bank_size, num_heads=num_heads, pool_size=pool_size,
                             pool_kernel=pool_kernel, pool_stride=pool_stride,
                             pool_padding=pool_padding)
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
    # Sanity check for H=270, W=480 (i.e. 1080/4 x 1920/4)
    # Bottleneck shape after 2x stride-2: (N, 128, 67, 120)
    # Pool conv: kernel=(9,15), stride=(8,15), padding=(1,0) → (N, 128, 8, 8)
    bank_size = 10
    model = FastDVDnet(
        bank_size=bank_size,
        num_heads=4,
        pool_size=8,
        pool_kernel=(9, 15),
        pool_stride=(8, 15),
        pool_padding=(1, 0)
    )
    bank  = KVBank(bank_size=bank_size)
    model.eval()
    with torch.no_grad():
        for t in range(5):
            frame    = torch.randn(1, 3, 270, 480)
            s_map    = torch.full((1, 1, 270, 480), 0.02)
            l_map    = torch.rand(1, 1, 270, 480) * 0.5
            out = model(frame, s_map, l_map, bank)
            print(f"t={t}  bank_len={len(bank)}  out={out.shape}")
    print("Sanity check passed!")
