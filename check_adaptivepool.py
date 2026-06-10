import math
import torch
import torch.nn as nn


def adaptive_avg_boundaries(in_size, out_size):
    """Exact [start, end) region per output index — matches AdaptiveAvgPool2d."""
    starts = [math.floor(i * in_size / out_size) for i in range(out_size)]
    ends   = [math.ceil((i + 1) * in_size / out_size) for i in range(out_size)]
    return list(zip(starts, ends))


class CustomAvgPool(nn.Module):
    """
    Static, compiler-friendly replacement for nn.AdaptiveAvgPool2d(out_grid)
    at ONE fixed input size (H_in, W_in).

    Adaptive avg pooling is separable: average over a rectangle = pool rows
    then pool cols. Each axis becomes a constant (out x in) averaging matrix P
    with P[o, k] = 1/len(region_o) for input index k in output region o.
    Output = Ph @ x @ Pw^T  -> two static matmuls, no dynamic axes.
    """
    def __init__(self, in_hw, out_grid=8):
        super().__init__()
        H_in, W_in = in_hw
        self.register_buffer("Ph", self._avg_matrix(H_in, out_grid))  # (G, H_in)
        self.register_buffer("Pw", self._avg_matrix(W_in, out_grid))  # (G, W_in)

    @staticmethod
    def _avg_matrix(in_size, out_size):
        P = torch.zeros(out_size, in_size)
        for o, (s, e) in enumerate(adaptive_avg_boundaries(in_size, out_size)):
            P[o, s:e] = 1.0 / (e - s)
        return P

    def forward(self, x):
        # x: (N, C, H, W) -> (N, C, G, G)
        x = torch.einsum('gh,nchw->ncgw', self.Ph, x)   # pool rows
        x = torch.einsum('ncgw,fw->ncgf', x, self.Pw)   # pool cols
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W, G = 2, 64, 270, 480, 8

    x = torch.randn(N, C, H, W)

    ref    = nn.AdaptiveAvgPool2d(G)(x)
    custom = CustomAvgPool((H, W), G)(x)

    print(f"input          : {tuple(x.shape)}")
    print(f"AdaptiveAvgPool: {tuple(ref.shape)}")
    print(f"CustomAvgPool  : {tuple(custom.shape)}")

    abs_err = (ref - custom).abs()
    print(f"\nmax  abs error : {abs_err.max().item():.3e}")
    print(f"mean abs error : {abs_err.mean().item():.3e}")
    print(f"allclose(1e-5) : {torch.allclose(ref, custom, atol=1e-5)}")