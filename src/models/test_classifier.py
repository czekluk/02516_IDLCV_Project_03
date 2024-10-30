import torch
import torch.nn as nn

class TestClassifier(nn.Module):
    """CNN classifier for testing purposes"""
    def __init__(self, num_classes=10):
        super(TestClassifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1 - input size: 32x32x3
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 2 - input size: 16x16x64
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 3 - input size: 8x8x128
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x