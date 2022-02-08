import torch
import time
import pyautogui
from torch.distributions import Categorical
import Capture
import Monte_Carlo_tree as mcts
import pydirectinput

''' selection = Capture.item_selection()
       if selection == 1:
           time.sleep(10)
           pydirectinput.press("enter")
       elif selection == 2:
           items = Capture.selection()
           #mcts_worker.input(items)'''


def run_once(model,r_latest):

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
    mcts_worker = mcts.MCTS()
    mcts_worker.backup()
    s_list, a_list, r_list = list(), list(), list()
    while 1:
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

        setting = torch.tensor(setting).float().reshape(-1, 1, 1080, 1724)

        prob = model.pi(x=setting, softmax_dim=0)
        prob = torch.nan_to_num(prob,0.2)
        prob = Categorical(prob).sample().numpy()
        a_list.append(prob)

        if prob == 0:
            pydirectinput.press("up")
        elif prob == 1:
            pydirectinput.press("down")

        elif prob == 2:
            pydirectinput.press("left")
        elif prob == 3:
            pydirectinput.press("right")

        r_list.append(0)
    tl = time.time()
    r_list.pop()
    reward = (tl - ts) - r_latest
    r_list.append(reward/60)
    if mcts_worker.checkwork():
        mcts_worker.append_reward(int(reward/60))
    return s_list, a_list, r_list

