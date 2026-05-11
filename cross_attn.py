import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CreateKV(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        # d is the number of tokens (spatial points) we compress the skip connection into
        self.pool = nn.AdaptiveAvgPool2d(int(np.sqrt(d))) 
        self.get_k = nn.Linear(d, d)
        self.get_v = nn.Linear(d, d)

    def forward(self, skip_feat):
        # skip_feat: (N, C, H, W)
        # We flatten the spatial dimension to get tokens
        # Output context: (N, C, d)
        kv_pooled = self.pool(skip_feat).flatten(2) 
        k = self.get_k(kv_pooled)
        v = self.get_v(kv_pooled)
        return k, v

class CrossAttentionSkip(nn.Module):
    def __init__(self, ch, d=64):
        super().__init__()
        self.ch = ch
        self.d = d
        
        # Query projection (keeping spatial resolution)
        self.q_proj = nn.Conv2d(ch, ch, kernel_size=1)
        
        # Final projection to mix the attention results
        self.out_proj = nn.Conv2d(ch, ch, kernel_size=1)
        
        # Learnable scale parameter initialized to 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, up_feat, k, v):
        """
        up_feat: (N, C, H, W) - Full resolution from decoder
        k: (N, C, d)         - Compressed Key from encoder
        v: (N, C, d)         - Compressed Value from encoder
        """
        N, C, H, W = up_feat.shape
        
        # 1. Project and reshape Query: (N, C, H, W) -> (N, H*W, C)
        # We treat each pixel in the upblock as a query
        q = self.q_proj(up_feat).flatten(2).transpose(1, 2) 
        
        # 2. Reshape K and V: (N, C, d) -> (N, C, d) and (N, d, C)
        # K stays (N, C, d) to match (N, HW, C) @ (N, C, d)
        # V will be transposed later to (N, d, C)
        
        # 3. Calculate Attention Map: (N, HW, C) @ (N, C, d) -> (N, HW, d)
        # This determines which compressed token 'd' belongs at which pixel 'HW'
        scaling = C ** -0.5
        attn = torch.matmul(q, k) * scaling
        attn = F.softmax(attn, dim=-1)
        
        # 4. Apply Attention to Values: (N, HW, d) @ (N, d, C) -> (N, HW, C)
        out = torch.matmul(attn, v.transpose(1, 2))
        
        # 5. Restore Spatial Shape: (N, HW, C) -> (N, C, H, W)
        out = out.transpose(1, 2).view(N, C, H, W)
        out = self.out_proj(out)
        
        # 6. Residual summation
        return up_feat + self.gamma * out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = 128
    latent_tokens = 64 # This is 'd'

    # Initialize modules
    getKV = CreateKV(d=latent_tokens).to(device)
    cross_attn = CrossAttentionSkip(ch=channels, d=latent_tokens).to(device)

    # Simulated tensors
    # up_feat is the current tensor in the decoder
    # skip_feat is the tensor from the encoder we want to "compress"
    up_feat = torch.randn(1, channels, 540, 960).to(device)
    skip_feat = torch.randn(1, channels, 540, 960).to(device)

    # 1. Create compressed KV from skip connection (Save this in memory)
    k, v = getKV(skip_feat) 
    print(f"Compressed K shape: {k.shape}") # (1, 128, 64)

    # 2. Use cross-attention to bring context back to up_feat
    output = cross_attn(up_feat, k, v)
    
    print(f"Final output shape: {output.shape}") # (1, 128, 540, 960)