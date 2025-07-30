class ShuffleNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 83, device="cuda"):
        super().__init__()

        self.model = models.shufflenet_v2_x1_0(pretrained=True)

        # Replace the first conv layer if in_channels != 3
        if in_channels != 3:
            # ShuffleNet v2 first layer is in self.model.conv1[0]
            out_channels = self.model.conv1[0].out_channels
            self.model.conv1[0] = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1, bias=False
            )

        # Replace the classifier (which is a Linear layer)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.model.to(device)

    def forward(self, x):
        return self.model(x)
