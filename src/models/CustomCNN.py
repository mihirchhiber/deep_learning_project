import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_filters=(16, 32, 64, 128)):
        super().__init__()
        self.num_filters = num_filters
        fl1, fl2, fl3, fl4 = self.num_filters

        self.embed = nn.Sequential(
            self.conv3x3(1, fl1),
            nn.MaxPool2d((3, 3), (2, 2)),

            self.conv3x3(fl1, fl2),
            nn.MaxPool2d((2, 3), (2, 2)),

            self.conv3x3(fl2, fl3),
            nn.MaxPool2d((5, 6), (5, 6)),

            self.conv3x3(fl3, fl4),
            nn.MaxPool2d((6, 6), (5, 4)),

            nn.Flatten(),
            nn.Linear(6*fl4, 4*fl4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classify = nn.Linear(4*fl4, 10)

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