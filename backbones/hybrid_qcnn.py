from modules import *
import torch.nn as nn
import torch.nn.functional as F
class Hybrid_RQCNN(nn.Module):
  def __init__(self, kernel_size, depth, circuit_layers, device, method, num_classes=3):
    super(Hybrid_RQCNN, self).__init__()
    if method == "structure_position":
      self.rqcnn= SPQCNN(kernel_size=2, depth=4, device=device, circuit_layers=4*4, method=method)
    else:
      self.rqcnn = RQCNN(kernel_size=2, depth=4, device=device, circuit_layers=1, method=method)
    self.pool = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(13*13*4, 64)
    self.fc2 = nn.Linear(64,num_classes)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    x = self.rqcnn(x)
    x = self.pool(x)
    #print(x.shape)
    x = x.view(-1, 13*13*4)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    x = self.sigmoid(x)
    return x
