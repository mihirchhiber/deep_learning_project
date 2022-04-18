import torch
import torch.nn as nn

class CustomCNN(nn.Module):
  def __init__(self):
    super().__init__()

    self._convblocks = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((3,3), (2,2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,3), (2,2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((5,6), (5,6)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((6,6), (5,4))
        )
    self._classifier = nn.Sequential(nn.Linear(in_features=1536, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(in_features=1024, out_features=10))
    # self._prob = nn.Softmax(dim=1)
    self.apply(self._init_weights)

  def forward(self, x):
      x = self._convblocks(x)
      x = x.view(x.size(0), -1)
      # out = self._classifier(x)
      # score = self._prob(out)
      score = self._classifier(x)
      return score

  def _init_weights(self, layer) -> None:
      if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)