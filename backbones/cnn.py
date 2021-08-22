import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, kernel_size, depth, num_classes=10):
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.depth = depth
        self.conv1 = nn.Conv2d(1, self.depth, kernel_size)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(13 * 13 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 13 * 13 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x