import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.conv1 = nn.Conv2d(1, 512, 20, stride=4)
        self.conv2 = nn.Conv2d(4, 16, 10, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=1)
        self.pi = nn.Linear(128, 4)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.relu(self.conv2(x))
        x = func.relu(self.conv3(x))
        return x

    def pi(self, x):
        x = self.forward(x)
        x = func.relu(self.pi(x.view(x.size(0), -1)))
        return x

    def v(self, x):
        x = self.forward(x)
        x = func.relu(self.v(x.view(x.size(0), -1)))
        return x
net =A2C()
input =torch.randn(1,1,512,512)
out =net.pi(input)
