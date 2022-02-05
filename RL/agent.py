import torch
import time
import pyautogui
from torch.distributions import Categorical
import Capture
import Monte_Carlo_tree as mcts
import RL.model as module


def run_once(model):
    ts = time.time()
    mcts_worker = mcts.MCTS()
    reward = 0
    s_list, a_list, r_list = list(), list(), list()
    while 1:
        game_over = Capture.game_over()
        if game_over is True:
            break
        selection = Capture.item_selection()
        if selection == 1:
            time.sleep(10)
            pyautogui.press("enter")
        elif selection == 2:
            items = Capture.selection()
            mcts_worker.input(items)
        setting = Capture.capture_screen()
        s_list.append(setting)
        setting = torch.tensor(setting).float().reshape(-1, 1, 28, 28)

        prob = model.pi(x=setting, softmax_dim=0)

        prob = Categorical(prob).sample().numpy()
        a_list.append(prob)

        if prob == 0:
            pyautogui.press("up")
            time.sleep(0.1)
        elif prob == 1:
            pyautogui.press("down")
            time.sleep(0.1)
        elif prob == 2:
            pyautogui.press("left")
            time.sleep(0.1)
        elif prob == 3:
            pyautogui.press("right")
            time.sleep(0.1)

        tl = time.time()
        reward = (tl - ts) / 60
        r_list.append(reward)
    mcts_worker.append_reward(reward)
    return s_list, a_list, r_list
