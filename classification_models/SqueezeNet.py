
import torch.nn as nn
from torchvision import models

class SqueezeNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 83, device="cuda"):
        super().__init__()
        self.model = models.squeezenet1_0(pretrained=True)
        
        # If input channels != 3, replace the first conv layer
        if in_channels != 3:
            self.model.features[0] = nn.Conv2d(in_channels, 96, kernel_size=7, stride=2)
        
        # Modify final classifier conv2d layer (SqueezeNet uses conv instead of linear)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.model.num_classes = num_classes
        self.model.to(device)
        
    def forward(self, x):
        return self.model(x)
