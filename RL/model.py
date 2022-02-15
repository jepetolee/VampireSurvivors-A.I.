import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

import gc
class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 10, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3)
        self.conv3 = nn.Conv2d(20, 40, 5, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)
        self.in_size = 3640
        self.h1 = 288

        self.lstm = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)

    def reset_lstm(self, buf_size=None):

        with torch.no_grad():
            self.h_t1 = self.c_t1 = torch.zeros(buf_size, self.h1, device=self.lstm.weight_ih.device)
    def del_lstm(self):
        del self.c_t1
        del self.h_t1
        torch.cuda.empty_cache()
        gc.collect()
    def forward(self, x):
        x = func.relu(self.max_pool1(self.conv1(x)))
        x = func.relu(self.max_pool2(self.conv2(x)))
        x = func.relu(self.max_pool3(self.conv3(x)))
        x = x.view(-1, 3640)
        self.h_t1, self.c_t1 = self.lstm(x, (self.h_t1, self.c_t1))
        del x
        return self.h_t1


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.encoder = FeatureEncoder()
        self.p = nn.Linear(self.encoder.in_size, 5)
        self.v = nn.Linear(self.encoder.in_size, 1)

    def pi(self, x, softmax_dim=1):

        x= self.encoder(x)
        x = self.p(x)
        prob = func.softmax(x, dim=softmax_dim)
        return prob

    def set_recurrent_buffers(self, buf_size):
        self.encoder.reset_lstm(buf_size=buf_size)
    def del_dat(self):
        self.encoder.del_lstm()

    def value(self, x):
        x=self.encoder(x)
        x = self.v(x)
        return x
