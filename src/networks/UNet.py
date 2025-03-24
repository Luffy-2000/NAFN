import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """1DCNN：
    (conv1d => BN => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downsampling: maxpool => conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """One-dimensional UNet network structure"""
    def __init__(self, in_channels=1, out_channels=1, out_features_size=None, **kwargs):
        super().__init__()
        
        # Feature extraction size, if None, the default value of 200 is used
        self.out_features_size = int(out_features_size if out_features_size is not None else 200)
        
        # Calculate the number of basic channels
        n1 = 64
        self.filters = [n1, n1 * 2, n1 * 4]  # [64, 128, 256]
        
        # Encoder part
        self.inc = ConvBlock(in_channels, self.filters[0])
        self.down1 = Down(self.filters[0], self.filters[1])
        self.down2 = Down(self.filters[1], self.filters[2])
        
        # Decoder part
        self.up2 = Up(self.filters[2], self.filters[1])
        self.up1 = Up(self.filters[1], self.filters[0])
        
        # Output layer
        self.outc = nn.Conv1d(self.filters[0], out_channels, kernel_size=1, stride=1, padding=0)
        
        # Feature extraction layer
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.filters[2], self.out_features_size)
        )
        

    def forward(self, x):
        # Encode
        batch, num_channels, num_pkts, num_fields = x.shape

        x = x.reshape(batch, num_channels, num_pkts * num_fields)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # Decode
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        
        logits = logits.reshape(batch, num_channels, num_pkts, num_fields)
        return logits

    def extract_features(self, x):
        batch, num_channels, num_pkts, num_fields = x.shape
        x = x.reshape(batch, num_channels, num_pkts * num_fields)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        output = self.feature_extractor(x3)
        return output
        