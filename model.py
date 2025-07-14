import torch
import torch.nn as nn
from torchvision.models import resnet18

class SpatialAttentionFusion(nn.Module):
    """Spatial attention fusion module for multimodal feature integration.
    
    Combines RGB and thermal features using channel-wise attention mechanisms.
    Inspired by CBAM (Convolutional Block Attention Module) with simplified architecture.
    """
    def __init__(self, in_channels=1024, out_channels=512):
        super().__init__()
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()  # Activation to bound values between 0 and 1
        )
    
    def forward(self, rgb, thermal):
        """Fuses RGB and thermal features using learned spatial attention.
        
        Args:
            rgb (Tensor): RGB features (batch_size, channels, H, W)
            thermal (Tensor): Thermal features (batch_size, channels, H, W)
            
        Returns:
            Tensor: Fused feature representation
        """
        x = torch.cat([rgb, thermal], dim=1)  # Concatenate along channel dimension
        attention = self.conv_fusion(x)
        return thermal + attention * rgb  # Residual connection with attended features

class DualModel(nn.Module):
    """Dual-stream architecture for multimodal road detection.
    
    Processes RGB and thermal imagery through separate ResNet18 backbones,
    fuses features with spatial attention, and produces detection outputs.
    
    Args:
        num_classes (int): Number of detection classes (default: 1)
    """
    def __init__(self, num_classes=1):
        super().__init__()
        # RGB processing stream
        self.rgb_backbone = resnet18(pretrained=True)
        self.rgb_backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rgb_features = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        
        # Thermal processing stream
        self.thermal_backbone = resnet18(pretrained=True)
        self.thermal_backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.thermal_features = nn.Sequential(*list(self.thermal_backbone.children())[:-2])
        
        # Spatial attention fusion module
        self.saf = SpatialAttentionFusion(in_channels=1024, out_channels=512)
        
        # Detection head - outputs 5 parameters per class
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Feature reduction
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Further compression
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),   # Final feature reduction
            nn.ReLU(),
            nn.Conv2d(64, 5 * num_classes, kernel_size=1)   # 5 outputs per class (x, y, w, h, conf)
        )
        
    def forward(self, rgb, thermal):
        """Forward pass through dual-stream architecture.
        
        Args:
            rgb (Tensor): RGB input tensor (batch_size, 3, H, W)
            thermal (Tensor): Thermal input tensor (batch_size, 3, H, W)
            
        Returns:
            Tensor: Detection outputs (batch_size, 5, grid_h, grid_w)
        """
        # Extract features from both modalities
        rgb_feat = self.rgb_features(rgb)
        thermal_feat = self.thermal_features(thermal)
        
        # Fuse features with spatial attention
        fused = self.saf(rgb_feat, thermal_feat)  # Inspired by Wang et al.'s CBAM with reduced computational cost
        
        # Detection head for road damage predictions
        return self.detection_head(fused)