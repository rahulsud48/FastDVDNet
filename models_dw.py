import torch
import torch.nn as nn


class DSConv(nn.Module):
    """
    Depthwise separable convolution:
      depthwise 3x3 + pointwise 1x1
    followed by BN + ReLU after each stage
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(DSConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_ch, out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class InputCvBlock(nn.Module):
    """
    Input block:
      1) grouped 3x3 conv across each frame+noise group
      2) depthwise separable conv for cheaper feature mixing

    Input channels:
      num_in_frames * (3 + 1)
      = RGB + noise_map per frame
    """
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        mid_ch = num_in_frames * self.interm_ch

        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_channels=num_in_frames * 4,   # 3 RGB + 1 noise per frame
                out_channels=mid_ch,
                kernel_size=3,
                padding=1,
                groups=num_in_frames,
                bias=False
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            DSConv(mid_ch, out_ch, stride=1)
        )

    def forward(self, x):
        return self.convblock(x)


class CvBlock(nn.Module):
    """
    Two depthwise separable conv blocks
    """
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, out_ch, stride=1),
            DSConv(out_ch, out_ch, stride=1)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    """
    Downsampling block:
      DSConv with stride=2
      followed by another CvBlock
    """
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, out_ch, stride=2),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    """
    Upsampling block:
      feature refinement with CvBlock
      cheap channel expansion with 1x1 conv
      PixelShuffle for x2 upsampling
    """
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(
                in_ch,
                out_ch * 4,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    """
    Output head:
      DSConv for lightweight refinement
      final 3x3 projection to RGB residual
    """
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            DSConv(in_ch, in_ch, stride=1),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """
    Single denoising block for 3-frame input:
      in0, in1, in2, noise_map
    Predicts residual/noise for center frame and returns:
      denoised = in1 - predicted_noise
    """
    def __init__(self, num_input_frames=3, chs_lyr0=32, chs_lyr1=64, chs_lyr2=128):
        super(DenBlock, self).__init__()

        self.chs_lyr0 = chs_lyr0
        self.chs_lyr1 = chs_lyr1
        self.chs_lyr2 = chs_lyr2

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
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        # Concatenate each frame with the same noise_map
        # [in0, noise, in1, noise, in2, noise] => 12 channels total
        x = torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1)

        x0 = self.inc(x)
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)

        x2_up = self.upc2(x2)
        x1_up = self.upc1(x1 + x2_up)

        x_out = self.outc(x0 + x1_up)

        # Residual learning: predict noise/residual on middle frame
        x_out = in1 - x_out
        return x_out


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
        # x: [N, 9, H, W] because 3 RGB frames
        x0, x1, x2 = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))
        x = self.temp(x0, x1, x2, noise_map)
        return x