import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np

class A2C(nn.Module):
    def __init__(self):
        self