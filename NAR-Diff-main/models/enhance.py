import torch
import torch.nn as nn
import math
from einops import rearrange

class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(c // self.num_heads)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class Illumination_Enhance_Net(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_heads=4, bias=True):
        super(Illumination_Enhance_Net, self).__init__()
        self.initial_conv = nn.Conv2d(1, num_features, kernel_size=3, padding=1, bias=bias)
        self.self_attention = Self_Attention(dim=num_features, num_heads=num_heads, bias=bias)
        self.enhance_block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=bias)
        )
        self.final_conv = nn.Conv2d(num_features, 1, kernel_size=1, bias=bias)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x_gray = x.mean(dim=1, keepdim=True)
        x_initial = self.initial_conv(x_gray)
        attn_out = self.self_attention(x_initial)
        enhanced_features = self.enhance_block(attn_out + x_initial)
        enhanced_illumination = self.final_conv(enhanced_features)
        enhanced_illumination = enhanced_illumination + x_gray
        enhanced_illumination = torch.cat([enhanced_illumination for i in range(3)], dim=1)
        enhanced_illumination = torch.sigmoid(enhanced_illumination)
        return enhanced_illumination