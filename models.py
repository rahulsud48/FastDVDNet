"""
FastDVDnet with Conv2d replaced by single grouped/depthwise-style convs
(uses max valid grouping via gcd(in_ch, out_ch) to preserve shapes)
"""

import math
import torch
import torch.nn as nn



def grouped_conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
    """
    Use the maximum valid grouping while preserving the requested shape.
    - If out_ch is a multiple of in_ch, this becomes true depthwise-like
      (groups=in_ch, possibly with channel multiplier > 1)
    - Otherwise it becomes a grouped conv using gcd(in_ch, out_ch)
    """
    groups = math.gcd(in_ch, out_ch)
    return nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias
    )


class CvBlock(nn.Module):
    """(Grouped/Depthwise-style Conv2d => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            grouped_conv(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            grouped_conv(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """Grouped/Depthwise-style conv version"""
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30

        in_ch_0 = num_in_frames * (3 + 1)      # 3 RGB + 1 noise per frame
        out_ch_0 = num_in_frames * self.interm_ch

        self.convblock = nn.Sequential(
            grouped_conv(in_ch_0, out_ch_0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch_0),
            nn.ReLU(inplace=True),
            grouped_conv(out_ch_0, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """Downscale + (Grouped/Depthwise-style Conv2d => BN => ReLU)*2"""
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            grouped_conv(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """(Grouped/Depthwise-style Conv2d => BN => ReLU)*2 + Upscale"""
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            grouped_conv(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    """Grouped/Depthwise-style Conv2d => BN => ReLU => Grouped/Depthwise-style Conv2d"""
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            grouped_conv(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            grouped_conv(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """
    Definition of the denoising block of FastDVDnet.
    """
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

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)

        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)

        x = self.outc(x0 + x1)

        # Residual
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
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x, noise_map):
        # x: [N, 9, H, W] because 3 RGB frames
        x0, x1, x2 = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))
        x = self.temp(x0, x1, x2, noise_map)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = FastDVDnet()

    summary(
        model,
        input_data=(torch.randn(1, 9, 96, 96), torch.randn(1, 1, 96, 96)),
        col_names=["input_size", "output_size", "num_params"],
        depth=5
    )