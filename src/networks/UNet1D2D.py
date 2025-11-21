import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv = ResidualConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv = ResidualConvBlock(in_channels, out_channels) 

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet1D2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=16, **kwargs):
        super().__init__()
        self.out_features_size = kwargs.get('out_features_size', 320)
        self.inc = ResidualConvBlock(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4) 
        self.up1 = Up(base_filters * 4, base_filters * 2)
        self.up2 = Up(base_filters * 2, base_filters)
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)


    def forward(self, x):
        # Extract features and intermediate features
        features, x1, x2, x3 = self.extract_features(x)
        # Reconstruct from features
        out = self.recon_features(features, x1, x2, x3)
        return out, features

    def extract_features(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        features = x3.flatten(1)
        return features, x1, x2, x3

    def recon_features(self, features, x1, x2, x3):
        # Reconstruct x3 from features
        x3_recon = features.view(x3.shape)
        # x3_recon = features.view(features.size(0), features.size(1), 1, 1)  # [B, C, 1, 1]
        # x3_recon = F.interpolate(x3_recon, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=False)
        # Upsampling path
        x = self.up1(x3_recon, x2)  # (10, 3)
        x = self.up2(x, x1)   # (20, 6)
        out = self.outc(x)    # output
        return out


