"""
Lightweight FastDVDnet experiment:
- Mostly 1x1 convolutions to reduce parameters
- No spatial downsampling / upsampling
- Shapes stay consistent
- Keeps original 5-frame, 2-stage FastDVDnet outer structure

This is an experiment only. Since 1x1 convs do not capture spatial neighborhoods,
denoising quality may drop compared to the original architecture.
"""

import torch
import torch.nn as nn


class CvBlock(nn.Module):
    """(1x1 Conv => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    """
    Input block:
    grouped 1x1 conv across each frame+noise group, then 1x1 conv
    Input channels = num_in_frames * (3 + 1)
    """
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        mid_ch = num_in_frames * self.interm_ch

        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_channels=num_in_frames * (3 + 1),
                out_channels=mid_ch,
                kernel_size=1,
                padding=0,
                groups=num_in_frames,
                bias=False
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=mid_ch,
                out_channels=out_ch,
                kernel_size=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """
    No actual downsampling in this lightweight experiment.
    Just channel transform + refinement at the same resolution.
    """
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """
    No actual upsampling in this lightweight experiment.
    Just same-resolution refinement.
    """
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    """1x1 Conv => BN => ReLU => 1x1 Conv"""
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """
    Lightweight denoising block.
    Same-resolution processing only.
    """
    def __init__(self, num_input_frames=3):
        super(DenBlock, self).__init__()

        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)

        # These are no longer true upsampling blocks; just same-resolution channel reducers
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)

        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        """
        in0, in1, in2: [N, 3, H, W]
        noise_map:     [N, 1, H, W]
        """
        x_in = torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1)

        x0 = self.inc(x_in)         # [N, 32, H, W]
        x1 = self.downc0(x0)        # [N, 64, H, W]
        x2 = self.downc1(x1)        # [N, 128, H, W]

        x2r = self.upc2(x2)         # [N, 64, H, W]
        x1r = self.upc1(x1 + x2r)   # [N, 32, H, W]

        x = self.outc(x0 + x1r)     # [N, 3, H, W]

        # Residual prediction on the center frame
        x = in1 - x
        return x


class FastDVDnet(nn.Module):
    """
    Original 5-frame, 2-stage outer FastDVDnet structure,
    but with lightweight same-resolution DenBlocks inside.
    """
    def __init__(self, num_input_frames=5):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = num_input_frames

        self.temp1 = DenBlock(num_input_frames=3)
        self.temp2 = DenBlock(num_input_frames=3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x, noise_map):
        """
        x: [N, num_frames*3, H, W]
        noise_map: [N, 1, H, W]
        """
        if self.num_input_frames != 5:
            raise ValueError(
                f"This model expects num_input_frames=5, got {self.num_input_frames}"
            )

        x0, x1, x2, x3, x4 = tuple(
            x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames)
        )

        # Stage 1
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)

        # Stage 2
        x = self.temp2(x20, x21, x22, noise_map)

        return x


if __name__ == "__main__":
    model = FastDVDnet()
    x = torch.randn(2, 15, 96, 96)
    noise_map = torch.randn(2, 1, 96, 96)
    y = model(x, noise_map)
    print(model)
    print("Output shape:", y.shape)