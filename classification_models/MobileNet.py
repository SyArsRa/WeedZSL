import torch.nn as nn
from torchvision import models

class MobileNetV2(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 83, device="cuda"):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        
        # MobileNetV2 does not support changing input channels easily, but you can override features[0][0]
        if in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify classifier to new num_classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.model.to(device)
        
    def forward(self, x):
        return self.model(x)
