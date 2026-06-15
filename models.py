"""
FastDVDnet — Single-frame input with KV Bank cross-attention at bottleneck.

Changes from uploaded version:
  - InputCvBlock now accepts 5 channels: RGB(3) + sigma_read(1) + lambda_shot(1)
  - Everything else identical to the uploaded working version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class learningBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(learningBlock, self).__init__()
        self.middle_channels = out_channels
        self.learning_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.learning_block(x)


class UpBlock(nn.Module):
    """CvBlock => ConvTranspose2d x2"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.Upscale_Block = nn.Sequential(
            learningBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.upsample(self.Upscale_Block(x))




class InputCvBlock(nn.Module):
    """
    First encoder block.
    Accepts 5 channels: RGB(3) + sigma_read map(1) + lambda_shot map(1).
    """
    def __init__(self, out_channels):
        super(InputCvBlock, self).__init__()
        # self.interm_ch = 30
        # 3 (RGB) + 1 (sigma_read) + 1 (lambda_shot) = 5 input channels
        self.Input_Cv_Block = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.Input_Cv_Block(x)


class DownBlock(nn.Module):
    """Stride-2 Conv2d => BN => ReLU => CvBlock"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            learningBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.convblock(x)



class OutputCvBlock(nn.Module):
    """Conv2d 3x3 => BN => ReLU => Conv2d 3x3"""
    def __init__(self, in_channels, out_channels):
        super(OutputCvBlock, self).__init__()
        self.Output_Cv_Block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.Output_Cv_Block(x)


# class KVBank:
#     def __init__(self, bank_size: int = 10, detach: bool = True):
#         self.bank_size = bank_size
#         self.detach    = detach
#         self._keys:   list = []
#         self._values: list = []

#     def reset(self):
#         self._keys.clear()
#         self._values.clear()

#     def push(self, k: torch.Tensor, v: torch.Tensor):
#         self._keys.append(k.detach() if self.detach else k)
#         self._values.append(v.detach() if self.detach else v)
#         if len(self._keys) > self.bank_size:
#             self._keys.pop(0)
#             self._values.pop(0)

#     def get(self):
#         if not self._keys:
#             return None, None
#         return torch.cat(self._keys, dim=1), torch.cat(self._values, dim=1)

#     def __len__(self):
#         return len(self._keys)

class AdaptiveAvgPool(nn.Module):
    def __init__(self, train_mode = True):
        super().__init__()
        # Initialize with training dimensions by default (96x96 -> 8x8)
        if train_mode:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=3)
        else:
            self.pool = nn.AvgPool2d(kernel_size=(31, 53), stride=(34, 61))
        
    # def train(self, mode=True):
    #     """Automatically adjusts pooling parameters when switching modes."""
    #     super().train(mode)
    #     if mode:
    #         # Training mode: expects 96x96 patch input
    #         self.pool = nn.AvgPool2d(kernel_size=12, stride=12)
    #     else:
    #         # Inference mode: expects 270x480 input for the custom compiler
    #         self.pool = nn.AvgPool2d(kernel_size=(31, 53), stride=(34, 61))
    #     return self

    def forward(self, x):
        # Your convolutional layers would go here
        x = self.pool(x)
        return x


class BottleneckCrossAttn(nn.Module):
    def __init__(self, ch: int = 128, num_heads: int = 4, pool_size: int = 8, train_mode = True):
        super(BottleneckCrossAttn, self).__init__()
        # self.pool = nn.AdaptiveAvgPool2d(pool_size)
        # self.pool = nn.AvgPool2d(kernel_size=12, stride=12)
        self.pool = AdaptiveAvgPool(train_mode)
        # self.pool = nn.Conv2d(
        #     in_channels=ch,
        #     out_channels=ch,
        #     kernel_size = (46,60),
        #     stride = (32,60),
        #     padding = (0,0),
        #     groups = ch,
        #     bias = False
        # )

        self.to_q = nn.Linear(ch, ch, bias=False)
        self.to_k = nn.Linear(ch, ch, bias=False)
        self.to_v = nn.Linear(ch, ch, bias=False)
        # self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(ch, ch), nn.Sigmoid())

    def _to_tokens(self, feat):
        return self.pool(feat).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, bank_k: torch.Tensor, bank_v: torch.Tensor):
        N, C, H, W = x.shape
        tokens = self._to_tokens(x)
        q_cur  = self.to_q(tokens)
        k_cur  = self.to_k(tokens)
        v_cur  = self.to_v(tokens)

        # bank_k, bank_v = bank.get()
        # if bank_k is None:
        #     return x, k_cur, v_cur

        # attn_out, _ = self.attn(q_cur, bank_k, bank_v)
        qkT = torch.matmul(q_cur, bank_k.transpose(-2,-1))/8.0
        attn_map = F.softmax(qkT, dim=-1)
        attn_out = torch.matmul(attn_map, bank_v)

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
    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8, train_mode = True):
        super(DenBlock, self).__init__()
        self.channels_layer0 = 16
        self.channels_layer1 = 16
        self.channels_layer2 = 32

        self.input_conv_block    = InputCvBlock(out_channels=self.channels_layer0)   # Y-ch input
        self.downsample0 = DownBlock(in_channels=self.channels_layer0, out_channels=self.channels_layer1)
        self.downsample1 = DownBlock(in_channels=self.channels_layer1, out_channels=self.channels_layer2 - 4)
        self.kv_attn = BottleneckCrossAttn(ch=self.channels_layer2,
                                           num_heads=num_heads,
                                           pool_size=pool_size, train_mode=train_mode)
        self.upsample2 = UpBlock(in_channels=self.channels_layer2, out_channels=self.channels_layer1)
        self.upsample1 = UpBlock(in_channels=self.channels_layer1, out_channels=self.channels_layer0)
        self.output_conv_block_y = OutputCvBlock(in_channels=self.channels_layer0, out_channels=1)
        self.output_conv_block_uv = OutputCvBlock(in_channels=self.channels_layer2, out_channels=2)
        # self.reset_params()

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
    #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # def reset_params(self):
    #     for _, m in enumerate(self.modules()):
    #         self.weight_init(m)

    def forward(self, x, uv_noise, bank_k, bank_v):
        # Concat: RGB + sigma_read + lambda_shot = 5 channels
        x0 = self.input_conv_block(x)
        x1 = self.downsample0(x0)
        x2 = self.downsample1(x1)
        x2 = torch.cat((x2, uv_noise), dim = 1)

        x2, k_cur, v_cur = self.kv_attn(x2, bank_k, bank_v)
        uv_res  = self.output_conv_block_uv(x2)

        x2 = self.upsample2(x2)
        x1 = self.upsample1(x1 + x2)
        x  = self.output_conv_block_y(x0 + x1)

        return x, uv_res, k_cur, v_cur


class FastDVDnet(nn.Module):
    def __init__(self, bank_size: int = 10, num_heads: int = 4, pool_size: int = 8, train_mode = False):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = 1
        self.temp = DenBlock(bank_size=bank_size, num_heads=num_heads, pool_size=pool_size, train_mode = train_mode)
        # self.reset_params()

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
    #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # def reset_params(self):
    #     for _, m in enumerate(self.modules()):
    #         self.weight_init(m)

    def forward(self, input_data, uv_noise, bank_k, bank_v):
        return self.temp(input_data, uv_noise, bank_k, bank_v)


if __name__ == "__main__":
    bank_size = 10
    model = FastDVDnet(bank_size=bank_size, num_heads=1, pool_size=8, train_mode=False)
    bank_k = torch.randn(1,10*64,32)
    bank_v = torch.randn(1,10*64,32)
    # bank  = KVBank(bank_size=bank_size)
    model.eval()
    H = 1080
    W = 1920
    # H = 96
    # W = 96
    with torch.no_grad():
        for t in range(12):
            y_frame    = torch.randn(1, 1, H, W)
            uv_frame    = torch.randn(1, 2, H//4, W//4)
            s_map    = torch.full((1, 1, H//4, W//4), 0.02)
            l_map    = torch.rand(1, 1, H//4, W//4) * 0.5
            uv_noise_input = torch.cat((uv_frame, s_map, l_map), dim = 1)
            y_res, uv_res, curr_k, curr_v = model(y_frame, uv_noise_input, bank_k, bank_v)
            bank_k = torch.cat([bank_k[:, 64:, :], curr_k], dim=1)
            bank_v = torch.cat([bank_v[:, 64:, :], curr_v], dim=1)
            print(f"t={t}  bank_k={bank_k.shape}  y_res={y_res.shape} uv_res={uv_res.shape}")
    print("Sanity check passed!")
