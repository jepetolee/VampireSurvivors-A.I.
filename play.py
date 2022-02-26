import torch
import RL.agent as agent
import RL.model as module
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gc
import time


def run():
    model = module.A2C()

    print("불러올 모델을 설정해주세요...:")
    model_name = input()
    model.load_state_dict(torch.load(model_name))

    print("CUDA:0으로 실행하시겠습니까?(Y/N):")
    gpu = input()

    if gpu == 'Y' or gpu == 'y':
        device = torch.device('cuda')
        print("use gpu to play this model...")

    else:
        device = torch.device('cpu')
        print("use gpu to play this model...")

    agent.run_once(model, device)
