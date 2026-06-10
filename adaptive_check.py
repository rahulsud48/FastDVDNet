import torch
import torch.nn as nn

class CompilerFriendlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize with training dimensions by default (96x96 -> 8x8)
        self.pool = nn.AvgPool2d(kernel_size=12, stride=12)
        
    def train(self, mode=True):
        """Automatically adjusts pooling parameters when switching modes."""
        super().train(mode)
        if mode:
            # Training mode: expects 96x96 patch input
            self.pool = nn.AvgPool2d(kernel_size=12, stride=12)
        else:
            # Inference mode: expects 270x480 input for the custom compiler
            self.pool = nn.AvgPool2d(kernel_size=(31, 53), stride=(34, 61))
        return self

    def forward(self, x):
        # Your convolutional layers would go here
        x = self.pool(x)
        return x

# 1. Create the dummy input tensor with your fixed shape
input_tensor = torch.randn(1, 64, 270, 480)

# 2. Define the Adaptive Average Pooling layer
adaptive_pool = nn.AdaptiveAvgPool2d(8)

# 3. Define your Standard AvgPool2d approximation
kernel_size = (31, 53)
stride = (34, 61)
standard_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

# 4. Run the forward passes
output_adaptive = adaptive_pool(input_tensor)
output_standard = standard_pool(input_tensor)

# 5. Check and print the shapes
print(f"Input shape:           {input_tensor.shape}")
print(f"Adaptive Pool shape:   {output_adaptive.shape}")
print(f"Standard Pool shape:   {output_standard.shape}\n")

print(output_adaptive)
print(output_standard)

# 6. Check the mathematical difference
# Calculate the maximum absolute difference between any two corresponding elements
max_diff = torch.max(torch.abs(output_adaptive - output_standard)).item()
print(f"Maximum absolute element-wise difference: {max_diff:.6f}")

# Check if they are close within a reasonable tolerance
are_close = torch.allclose(output_adaptive, output_standard, atol=1e-2)
print(f"Are the outputs mathematically close? {are_close}")