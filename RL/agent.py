import time
import torch
import pyautogui
from torch.distributions import Categorical
import Capture
import Monte_Carlo_tree as mcts
import pydirectinput
import gc


def run_once(model, device):
    pydirectinput.FAILSAFE = False
    pyautogui.FAILSAFE = False

    mcts_worker = mcts.MCTS()
    mcts_worker.backup()

    s_list, a_list, r_list, p_list, mcts_list = list(), list(), list(), list(), list()

    model.to(device)
    model.set_recurrent_buffers(buf_size=1)

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

    time.sleep(1.5)

    probal = 1
    pydirectinput.keyDown("down")

    reward_sum = 0
    setting, mcts_tensor, result = Capture.item_selection(mcts_worker)

    with torch.no_grad():

        while result >= 0:

            s_list.append(setting)
            mcts_list.append(mcts_tensor)

            mcts_setting = torch.tensor(mcts_tensor).float().reshape(-1, 3000).to(device)
            setting = torch.tensor(setting).float().reshape(-1, 1, 1080, 1724).to(device)

            prob = model.pi(x=setting, mcts_setting=mcts_setting, softmax_dim=1)
            prob = prob.view(-1)

            del setting
            torch.cuda.empty_cache()

            a = Categorical(prob).sample().to('cpu')
            p_list.append(prob[a].item())
            a = a.numpy()
            a_list.append(a)

            if probal == 0:
                pydirectinput.keyUp("up")
            elif probal == 1:
                pydirectinput.keyUp("down")
            elif probal == 2:
                pydirectinput.keyUp("left")
            elif probal == 3:
                pydirectinput.keyUp("right")

            probal = a

            if a == 0:
                pydirectinput.keyDown("up")
            elif a == 1:
                pydirectinput.keyDown("down")
            elif a == 2:
                pydirectinput.keyDown("left")
            elif a == 3:
                pydirectinput.keyDown("right")

            setting, mcts_tensor, result = Capture.item_selection(mcts_worker)

            if result < 0:

                for i in range(5):
                    reward_sum += result - 3 * i
                    r_list[-4 + i] += (result - 3 * i)

                reward = 0

            elif result > 0:
                for i in range(10):
                    r_list[-9 + i] += (result + i)

                reward = 0

            else:
                reward = 1

            reward_sum += reward
            r_list.append(reward)

        if mcts_worker.checkwork():
            mcts_worker.append_reward(int(reward_sum / 10))
            mcts_worker.save()

        gc.collect()



    return s_list, a_list, r_list, p_list, mcts_list
