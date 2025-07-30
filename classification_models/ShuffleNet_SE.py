
from torch import nn
import torch

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

# SE Attention Module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Shuffle Unit with ONLY standard Conv (No Separable Conv)
class ShuffleUnit(nn.Module):
    def __init__(self, inp, outp, stride):
        super().__init__()
        self.stride = stride
        mid = outp // 2

        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
                nn.BatchNorm2d(mid),

                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, mid, 3, stride, 1, bias=False),
                nn.BatchNorm2d(mid),

                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid, mid, 3, stride, 1, bias=False),
                nn.BatchNorm2d(mid),

                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)

# Final Model (SE + no Separable Conv)
class ShuffleNetV2WithSE(nn.Module):
    def __init__(self, num_classes=83, repeats=[4, 8, 4], channels=[24, 116, 232, 464, 1024], se_reduction=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.stage2 = self._make_stage(channels[0], channels[1], repeats[0])
        self.stage3 = self._make_stage(channels[1], channels[2], repeats[1])
        self.stage4 = self._make_stage(channels[2], channels[3], repeats[2])

        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
        )

        self.se = SELayer(channels[4], reduction=se_reduction)

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[4], num_classes)

        self._initialize_weights()
        
    def _make_stage(self, inp, outp, repeats):
        layers = [ShuffleUnit(inp, outp, 2)]
        layers += [ShuffleUnit(outp, outp, 1) for _ in range(repeats - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.se(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
