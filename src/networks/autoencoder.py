import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils import get_output_dim
from networks.lopez17cnn import Lopez17CNN
import math

class Decoder(nn.Module):
    """解码器模块，用于将编码特征重建为原始输入"""
    def __init__(self, in_features, out_channels, target_size, filters):
        super().__init__()
        self.target_size = target_size  # (H, W) - 目标输出尺寸
        self.filters = filters
        
        # 计算初始特征图大小
        self.initial_size = (target_size[0]//4, target_size[1]//2)  # 修改初始宽度为目标的1/2
        
        self.layers = nn.ModuleList([
            # 全连接层将编码向量转换为特征图
            nn.Sequential(
                nn.Linear(in_features, self.initial_size[0] * self.initial_size[1] * filters[1]),
                nn.BatchNorm1d(self.initial_size[0] * self.initial_size[1] * filters[1]),
                nn.ReLU()
            ),
            
            # 第一层上采样卷积 - 高度x2
            nn.Sequential(
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
                nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(filters[0]),
                nn.ReLU()
            ),
            
            # 第二层上采样卷积 - 高度x2，宽度x2
            nn.Sequential(
                nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
                nn.Conv2d(filters[0], filters[0]//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters[0]//2),
                nn.ReLU()
            ),
            
            # 最终输出层
            nn.Sequential(
                nn.Conv2d(filters[0]//2, out_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        ])
        
    def forward(self, x):
        # 全连接层重塑
        batch_size = x.size(0)
        out = self.layers[0](x)
        out = out.view(batch_size, self.filters[1], self.initial_size[0], self.initial_size[1])
        
        # 上采样卷积层
        for layer in self.layers[1:]:
            out = layer(out)
            
        return out

class Autoencoder(nn.Module):
    """去噪自编码器网络，使用Lopez17CNN作为编码器，非对称解码器用于无监督预训练"""
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()
        
        num_pkts = kwargs['num_pkts']
        num_fields = kwargs['num_fields']
        self.out_features_size = kwargs['out_features_size'] or 200
        
        # 使用Lopez17CNN作为编码器
        self.encoder = Lopez17CNN(
            in_channels=in_channels,
            num_pkts=num_pkts,
            num_fields=num_fields,
            out_features_size=self.out_features_size,
            scale=kwargs.get('scale', 1)
        )
        
        # 计算滤波器大小
        scaling_factor = kwargs.get('scale', 1)
        self.filters = [math.ceil(32 * scaling_factor), math.ceil(64 * scaling_factor)]
        
        # 创建解码器
        self.decoder = Decoder(
            in_features=self.out_features_size,
            out_channels=in_channels,
            target_size=(num_pkts, num_fields),  # 目标输出尺寸
            filters=self.filters
        )
        
        
    def add_noise(self, x):
        """添加高斯噪声"""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise
        
    def encode(self, x):
        """编码过程"""
        return self.encoder.extract_features(x)
        
    def decode(self, x):
        """解码过程"""
        return self.decoder(x)
    
    def forward(self, x):
        # print(f"Debug - x_noisy shape: {x_noisy.shape}")
        
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded
    
    def extract_features(self, x):
        """提取特征用于下游任务"""
        return self.encoder.extract_features(x)