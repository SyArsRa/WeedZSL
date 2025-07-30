import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 83, device="cuda"):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        
        # If input channels != 3, adjust first conv layer (optional, if you need)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fc layer for num_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(device)
        
    def forward(self, x):
        return self.model(x)
