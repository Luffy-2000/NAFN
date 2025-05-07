import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils import get_output_dim
from networks.lopez17cnn import Lopez17CNN
import math

class Decoder(nn.Module):
    """Decoder module to reconstruct original input from encoded features"""
    def __init__(self, in_features, out_channels, target_size, filters):
        super().__init__()
        self.target_size = target_size  # (H, W)
        self.filters = filters
        
        self.initial_size = (target_size[0]//4, target_size[1]//2)  # Modify the initial width to 1/2 of the target
        
        self.layers = nn.ModuleList([
            # The fully connected layer converts the encoded vector into a feature map
            nn.Sequential(
                nn.Linear(in_features, self.initial_size[0] * self.initial_size[1] * filters[1]),
                nn.BatchNorm1d(self.initial_size[0] * self.initial_size[1] * filters[1]),
                nn.ReLU()
            ),
            
            # First layer of upsampling convolution - height x2
            nn.Sequential(
                nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True),
                nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(filters[0]),
                nn.ReLU()
            ),
            
            # Second layer of upsampling convolution - height x2, width x2
            nn.Sequential(
                nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True),
                nn.Conv2d(filters[0], filters[0]//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters[0]//2),
                nn.ReLU()
            ),
            
            nn.Sequential(
                nn.Conv2d(filters[0]//2, out_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        ])
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.layers[0](x)
        out = out.view(batch_size, self.filters[1], self.initial_size[0], self.initial_size[1])
        
        for layer in self.layers[1:]:
            out = layer(out)
            
        return out

class Autoencoder(nn.Module):
    """Denoising autoencoder network using Lopez17CNN as encoder, asymmetric decoder for unsupervised pre-training"""
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()
        
        num_pkts = kwargs['num_pkts']
        num_fields = kwargs['num_fields']
        self.out_features_size = kwargs['out_features_size'] or 200
        
        # Use Lopez17CNN as encoder
        self.encoder = Lopez17CNN(
            in_channels=in_channels,
            num_pkts=num_pkts,
            num_fields=num_fields,
            out_features_size=self.out_features_size,
            scale=kwargs.get('scale', 1)
        )

        scaling_factor = kwargs.get('scale', 1)
        self.filters = [math.ceil(32 * scaling_factor), math.ceil(64 * scaling_factor)]
        
        self.decoder = Decoder(
            in_features=self.out_features_size,
            out_channels=in_channels,
            target_size=(num_pkts, num_fields),  
            filters=self.filters
        )
        
        
    def add_noise(self, x):
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise
        
    def encode(self, x):
        """Encoding process"""
        return self.encoder.extract_features(x)
        
    def decode(self, x):
        """Decoding process"""
        return self.decoder(x)
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded
    
    def extract_features(self, x):
        """Extract features for downstream tasks"""
        return self.encoder.extract_features(x)


