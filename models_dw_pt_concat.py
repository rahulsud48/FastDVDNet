"""
Definition of the FastDVDnet model
Depthwise-separable version:
each Conv2d(3x3) is replaced by:
    depthwise 3x3 + pointwise 1x1
No no_orthog changes introduced.
"""

import torch
import torch.nn as nn
import time

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
            ChannelToSpace(scale=2)
            # nn.PixelShuffle(2)
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
        self.upc1 = UpBlock(in_ch=self.chs_lyr1*2, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0*2, out_ch=3)

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
        # x1 = self.upc1(x1 + x2)
        x1 = self.upc1(torch.cat((x1, x2), dim = 1))

        # x = self.outc(x0 + x1)
        x = self.outc(torch.cat((x0, x1), dim = 1))

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