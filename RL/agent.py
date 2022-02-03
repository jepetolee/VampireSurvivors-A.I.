import torch
import time
import pyautogui
from torch.distributions import Categorical
import Capture
import RL.model as module


def run_once(model):
    ts = time.time()
    game_over = False
    while not game_over:
        env = Capture.capture_screen()
        env = torch.from_numpy(env).float()
        env = torch.reshape(env, (1, 1, 28, 28))
        prob = model.pi(x=env, softmax_dim=0)
        prob = Categorical(prob).sample().numpy()
        if prob == 0:
            pyautogui.press("up")
        elif prob == 1:
            pyautogui.press("down")
        elif prob == 2:
            pyautogui.press("left")
        elif prob == 3:
            pyautogui.press("right")
        game_over = Capture.game_over()
    tl = time.time()

    return (tl - ts) / 10