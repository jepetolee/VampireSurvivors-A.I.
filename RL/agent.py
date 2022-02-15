import random
import torch
import time
import pyautogui
from torch.distributions import Categorical
import Capture
import Monte_Carlo_tree as mcts
import pydirectinput
import gc

def run_once(model,r_latest):
    pydirectinput.FAILSAFE = False
    mcts_worker = mcts.MCTS()
    device = torch.device('cuda')
    mcts_worker.backup()
    reward = 0
    s_list, a_list, r_list = list(), list(), list()
    model.set_recurrent_buffers(buf_size=1)
    time.sleep(4)
    pyautogui.moveTo(980, 660)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.moveTo(1230, 1000)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.moveTo(1230, 1000)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.moveTo(740, 300)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.moveTo(1250, 1000)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.moveTo(1250, 1000)
    pyautogui.click()

    ts = time.time()
    probal =1
    with torch.no_grad():
        while 1:
            if probal == 0:
                pydirectinput.keyUp("up")
            elif probal == 1:
                pydirectinput.keyUp("down")
            elif probal == 2:
                pydirectinput.keyUp("left")
            elif probal == 3:
                pydirectinput.keyUp("right")
            game_over = Capture.game_over()

            if game_over is True:
                time.sleep(3)
                pyautogui.moveTo(950, 750)
                pyautogui.click()
                time.sleep(0.5)
                pyautogui.moveTo(960, 1000)
                pyautogui.click()
                time.sleep(0.5)
                pyautogui.click()
                time.sleep(0.5)
                pyautogui.click()
                break

            Capture.item_selection(mcts_worker)

            setting = Capture.capture_screen()
            s_list.append(setting)

            setting = torch.tensor(setting).float().reshape(-1, 1, 1080, 1724).to(device)

            prob = model.pi(x=setting, softmax_dim=1)
            del setting
            torch.cuda.empty_cache()
            prob = Categorical(prob).sample().to('cpu').numpy()

            a_list.append(prob)

            probal = prob

            if prob == 0:
                pydirectinput.keyDown("up")
            elif prob == 1:
                pydirectinput.keyDown("down")
            elif prob == 2:
                pydirectinput.keyDown("left")
            elif prob == 3:
                pydirectinput.keyDown("right")
            tl = time.time()
            reward = tl - ts - r_latest
            r_list.append(reward)

    if mcts_worker.checkwork():
        mcts_worker.append_reward(int(reward/10))
    gc.collect()
    mcts_worker.save()
    return s_list, a_list, r_list
