"""
UNet-style FastDVDnet model
High parameter count, standard Conv2d, no depthwise separable optimizations.
Same input/output interface as original FastDVDnet.
"""

import torch
import torch.nn as nn
import time


class ConvBlock(nn.Module):
    """Standard (Conv2d => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class InputBlock(nn.Module):
    """First block — takes concatenated frames + noise maps"""
    def __init__(self, num_in_frames, out_ch):
        super().__init__()
        in_ch = num_in_frames * (3 + 1)  # RGB + noise map per frame
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool downscale + ConvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose2d upscale + ConvBlock (takes skip connection via concat)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)  # *2 for skip concat

    def forward(self, x, skip):
        x = self.up(x)

        # Handle spatial mismatch due to odd H or W
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = torch.nn.functional.interpolate(
                x, 
                size=(skip.shape[2], skip.shape[3]),
                mode='bilinear', 
                align_corners=False
            )

        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class OutputBlock(nn.Module):
    """Final conv to produce 3 channel output"""
    def __init__(self, in_ch, out_ch=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


class DenBlock(nn.Module):
    """UNet-style denoising block with deep encoder-decoder + skip connections"""
    def __init__(self, num_input_frames=3):
        super().__init__()
        # Channel sizes — much larger than original
        c0  = 64
        c1  = 128
        c2  = 256
        c3  = 512
        c4  = 1024  # bottleneck

        # Encoder
        self.inc    = InputBlock(num_in_frames=num_input_frames, out_ch=c0)
        self.down1  = DownBlock(c0, c1)
        self.down2  = DownBlock(c1, c2)
        self.down3  = DownBlock(c2, c3)
        self.down4  = DownBlock(c3, c4)   # bottleneck

        # Decoder
        self.up1    = UpBlock(c4, c3)
        self.up2    = UpBlock(c3, c2)
        self.up3    = UpBlock(c2, c1)
        self.up4    = UpBlock(c1, c0)

        # Output
        self.outc   = OutputBlock(in_ch=c0, out_ch=3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        inp = torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1)

        # Encoder
        x0 = self.inc(inp)      # (N, 64,   H,    W)
        x1 = self.down1(x0)     # (N, 128,  H/2,  W/2)
        x2 = self.down2(x1)     # (N, 256,  H/4,  W/4)
        x3 = self.down3(x2)     # (N, 512,  H/8,  W/8)
        x4 = self.down4(x3)     # (N, 1024, H/16, W/16) bottleneck

        # Decoder with skip connections
        x  = self.up1(x4, x3)   # (N, 512,  H/8,  W/8)
        x  = self.up2(x,  x2)   # (N, 256,  H/4,  W/4)
        x  = self.up3(x,  x1)   # (N, 128,  H/2,  W/2)
        x  = self.up4(x,  x0)   # (N, 64,   H,    W)

        x  = self.outc(x)       # (N, 3,    H,    W)
        return x


class FastDVDnet(nn.Module):
    def __init__(self, num_input_frames=3):
        super().__init__()
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
        return self.temp(x0, x1, x2, noise_map)


if __name__ == "__main__":
    from torchinfo import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastDVDnetUNet().to(device)
    model.eval()

    input_data  = torch.randn(1, 9,  1080, 1920).to(device)
    noise_map   = torch.randn(1, 1,  1080, 1920).to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model(input_data, noise_map)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Device     :", device)
    print("Time taken :", end_time - start_time)
    print("Output shape:", output.shape)  # (1, 3, 1080, 1920)

    summary(
        model,
        input_data=(torch.randn(1, 9, 1080, 1920), torch.randn(1, 1, 1080, 1920)),
        col_names=["input_size", "output_size", "num_params"],
        depth=5
    )