import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            self.conv3x3(1, 16),
            nn.MaxPool2d((3, 3), (2, 2)),

            self.conv3x3(16, 32),
            nn.MaxPool2d((2, 3), (2, 2)),

            self.conv3x3(32, 64),
            nn.MaxPool2d((5, 6), (5, 6)),

            self.conv3x3(64, 128),
            nn.MaxPool2d((6, 6), (5, 4)),

            nn.Flatten(),
            nn.Linear(2*128, 128),
            nn.ReLU()
        )

        self.classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

        self.apply(self._init_weights)

    def conv3x3(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        embedding = self.embed(input)
        output = self.classify(embedding)
        return output

    def _init_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
