import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=4, embed_dim=64, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        return x

class MHSA(nn.Module):
    def __init__(self, dim, num_heads, window_size=8):
        super(MHSA, self).__init__()

        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        x = x.permute(1, 2, 0).reshape(B, C, H, W)  # (B, C, H, W)
        return x

class MultiScaleFeatures(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MultiScaleFeatures, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(input_channels, output_channels // 4, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(input_channels, output_channels // 4, kernel_size=7, stride=1, padding=3)
        self.fusion = nn.Conv2d(output_channels // 2 + output_channels // 4 + output_channels // 4, output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x = torch.cat([x1, x2, x3], dim=1)  # Concatenate along channel dimension
        x = self.fusion(x)
        return x


class GLSE(nn.Module):
    def __init__(self, input_channels=4, output_channels=32, img_size=128, embed_dim=64, patch_size=4, num_heads=4, num_mhsa=4):
        super(GLSE, self).__init__()

        self.high_res_features = MultiScaleFeatures(input_channels, output_channels)

        self.patch_embed = PatchEmbedding(in_channels=input_channels, embed_dim=embed_dim, patch_size=patch_size)

        self.mhsas = nn.ModuleList([MHSA(dim=embed_dim, num_heads=num_heads) for _ in range(num_mhsa)])

        self.second_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)

        self.conv = nn.Conv2d(embed_dim, output_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        high_res_features = self.high_res_features(x)

        x = self.patch_embed(x)

        for mhsa in self.mhsas:
            x = mhsa(x) + x 
        x = self.second_embed(x)

        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        x = self.conv(x)

        x = x + high_res_features
        
        return x


if __name__ == "__main__":
    model = GLSE(output_channels=10)
    input_tensor = torch.randn(1, 4, 128, 128)  # Batch size B=1, Channels=4, Height=128, Width=128
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output: (1, 32, 128, 128)
