import torch
import torch.nn as nn

# Channel Shuffle
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

# Separable Convolution (Depthwise + Pointwise)
class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activate=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = activate
        if activate:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if self.activate:
            x = self.relu(x)
        return x

# Shuffle Unit with Separable Conv (NO SE)
class ShuffleUnitSepConv(nn.Module):
    def __init__(self, inp, outp, stride):
        super().__init__()
        self.stride = stride
        mid = outp // 2

        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),

                SeparableConv(mid, mid, kernel_size=3, stride=1, padding=1, activate=False),

                nn.Conv2d(mid, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                SeparableConv(inp, inp, kernel_size=3, stride=stride, padding=1, activate=False),
                nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, mid, 1, 1, 0, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),

                SeparableConv(mid, mid, kernel_size=3, stride=stride, padding=1, activate=False),

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

# Final Model: Separable Convs only, No SE
class ShuffleNetV2WithSepConv(nn.Module):
    def __init__(self, num_classes=83, repeats=[4, 8, 4], channels=[24, 116, 232, 464, 1024]):
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

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[4], num_classes)


    def _make_stage(self, inp, outp, repeats):
        layers = [ShuffleUnitSepConv(inp, outp, 2)]
        layers += [ShuffleUnitSepConv(outp, outp, 1) for _ in range(repeats - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
