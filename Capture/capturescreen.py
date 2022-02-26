import time
import cv2 as cv
import numpy as np
import pyautogui


def item_selection(mcts):

    src = pyautogui.screenshot(region=(100, 0, 1724, 1080))
    src = np.array(src)
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

    dropbox = cv.imread("Capture/data/dropbox.png")
    dropbox = cv.cvtColor(dropbox, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, dropbox, cv.TM_SQDIFF)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    select = cv.imread("Capture/data/selection.png")
    select = cv.cvtColor(select, cv.COLOR_RGB2GRAY)
    result2 = cv.matchTemplate(src, select, cv.TM_SQDIFF)

    min_val1, max_val1, min_loc1, max_loc1 = cv.minMaxLoc(result2)

    revival = cv.imread("Capture/data/revival.png")
    revival = cv.cvtColor(revival, cv.COLOR_RGB2GRAY)
    resultR = cv.matchTemplate(src, revival, cv.TM_SQDIFF)

    min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(resultR)

    tempD = cv.imread("Capture/data/end.png")
    tempD = cv.cvtColor(tempD, cv.COLOR_RGB2GRAY)
    resultD = cv.matchTemplate(src, tempD, cv.TM_SQDIFF)

    min_valD, max_valD, min_locD, max_locD = cv.minMaxLoc(resultD)

    if min_valD < 14000000:

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
        mcts_tensor = mcts.tensor()

        return src, mcts_tensor, -75

    if min_val2 < 14000000:

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
        mcts_tensor = mcts.tensor()

        return src, mcts_tensor, -25

    if min_val < 15000000:

        pyautogui.moveTo(950, 880)
        pyautogui.click()
        time.sleep(20)
        pyautogui.moveTo(950, 880)
        pyautogui.click()

        mcts_tensor = mcts.tensor()
        return src, mcts_tensor, 20

    if min_val1 < 1468192:

        res = selection()
        result = mcts.input(res)

        if result == 3:
            pyautogui.moveTo(970, 840)
            pyautogui.click()
        elif result == 0:
            pyautogui.moveTo(970, 310)
            pyautogui.click()
        elif result == 2:
            pyautogui.moveTo(970, 650)
            pyautogui.click()
        else:
            pyautogui.moveTo(970, 480)
            pyautogui.click()

        mcts_tensor = mcts.tensor()
        return src, mcts_tensor, 15

    mcts_tensor = mcts.tensor()
    return src, mcts_tensor, 0


def selection():

    time.sleep(1.5)

    src = pyautogui.screenshot()
    src = np.array(src)
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

    temp1 = cv.imread("Capture/data/items/armor.png")
    temp1 = cv.cvtColor(temp1, cv.COLOR_RGB2GRAY)

    temp2 = cv.imread("Capture/data/items/attractorb.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)

    temp3 = cv.imread("Capture/data/items/axe.png")
    temp3 = cv.cvtColor(temp3, cv.COLOR_RGB2GRAY)

    temp4 = cv.imread("Capture/data/items/bone.png")
    temp4 = cv.cvtColor(temp4, cv.COLOR_RGB2GRAY)

    temp5 = cv.imread("Capture/data/items/bracer.png")
    temp5 = cv.cvtColor(temp5, cv.COLOR_RGB2GRAY)

    temp6 = cv.imread("Capture/data/items/candelabrador.png")
    temp6 = cv.cvtColor(temp6, cv.COLOR_RGB2GRAY)

    temp7 = cv.imread("Capture/data/items/clock_lancet.png")
    temp7 = cv.cvtColor(temp7, cv.COLOR_RGB2GRAY)

    temp8 = cv.imread("Capture/data/items/clover.png")
    temp8 = cv.cvtColor(temp8, cv.COLOR_RGB2GRAY)

    temp9 = cv.imread("Capture/data/items/cross.png")
    temp9 = cv.cvtColor(temp9, cv.COLOR_RGB2GRAY)

    temp10 = cv.imread("Capture/data/items/crown.png")
    temp10 = cv.cvtColor(temp10, cv.COLOR_RGB2GRAY)

    temp11 = cv.imread("Capture/data/items/duplicator.png")
    temp11 = cv.cvtColor(temp11, cv.COLOR_RGB2GRAY)

    temp12 = cv.imread("Capture/data/items/ebony_wings.png")
    temp12 = cv.cvtColor(temp12, cv.COLOR_RGB2GRAY)

    temp13 = cv.imread("Capture/data/items/empty_book.png")
    temp13 = cv.cvtColor(temp13, cv.COLOR_RGB2GRAY)

    temp14 = cv.imread("Capture/data/items/fire_wand.png")
    temp14 = cv.cvtColor(temp14, cv.COLOR_RGB2GRAY)

    temp15 = cv.imread("Capture/data/items/garlic.png")
    temp15 = cv.cvtColor(temp15, cv.COLOR_RGB2GRAY)

    temp16 = cv.imread("Capture/data/items/hollow_heart.png")
    temp16 = cv.cvtColor(temp16, cv.COLOR_RGB2GRAY)

    temp17 = cv.imread("Capture/data/items/king_bible.png")
    temp17 = cv.cvtColor(temp17, cv.COLOR_RGB2GRAY)

    temp18 = cv.imread("Capture/data/items/knife.png")
    temp18 = cv.cvtColor(temp18, cv.COLOR_RGB2GRAY)

    temp19 = cv.imread("Capture/data/items/laurel.png")
    temp19 = cv.cvtColor(temp19, cv.COLOR_RGB2GRAY)

    temp20 = cv.imread("Capture/data/items/lightning_ring.png")
    temp20 = cv.cvtColor(temp20, cv.COLOR_RGB2GRAY)

    temp21 = cv.imread("Capture/data/items/magic_wand.png")
    temp21 = cv.cvtColor(temp21, cv.COLOR_RGB2GRAY)

    temp22 = cv.imread("Capture/data/items/peachone.png")
    temp22 = cv.cvtColor(temp22, cv.COLOR_RGB2GRAY)

    temp23 = cv.imread("Capture/data/items/pummarola.png")
    temp23 = cv.cvtColor(temp23, cv.COLOR_RGB2GRAY)

    temp24 = cv.imread("Capture/data/items/runetracer.png")
    temp24 = cv.cvtColor(temp24, cv.COLOR_RGB2GRAY)

    temp25 = cv.imread("Capture/data/items/santa_water.png")
    temp25 = cv.cvtColor(temp25, cv.COLOR_RGB2GRAY)

    temp26 = cv.imread("Capture/data/items/spellbinder.png")
    temp26 = cv.cvtColor(temp26, cv.COLOR_RGB2GRAY)

    temp27 = cv.imread("Capture/data/items/spinach.png")
    temp27 = cv.cvtColor(temp27, cv.COLOR_RGB2GRAY)

    temp28 = cv.imread("Capture/data/items/whip.png")
    temp28 = cv.cvtColor(temp28, cv.COLOR_RGB2GRAY)

    temp29 = cv.imread("Capture/data/items/wings.png")
    temp29 = cv.cvtColor(temp29, cv.COLOR_RGB2GRAY)

    temp30 = cv.imread("Capture/data/items/pentagram.png")
    temp30 = cv.cvtColor(temp30, cv.COLOR_RGB2GRAY)

    case = []
    temp_all = [temp1, temp2, temp3, temp4, temp5,
                temp6, temp7, temp8, temp9, temp10,
                temp11, temp12, temp13, temp14, temp15,
                temp16, temp17, temp18, temp19, temp20,
                temp21, temp22, temp23, temp24, temp25,
                temp26, temp27, temp28, temp29, temp30]

    thresholds = [10000000, 10000000, 10000000, 10000000, 10000000,
                  10000000, 10000000, 10000000, 10000000, 10000000,
                  10000000, 10000000, 10000000, 10000000, 10000000,
                  10000000, 10000000, 10000000, 7000704, 10000000,
                  10000000, 10000000, 10000000, 10000000, 10000000,
                  10000000, 7019776, 10000000, 10000000, 10000000, ]
    adder = 0

    for temp in temp_all:

        result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if min_val < thresholds[adder]:
            case.append(adder)
        adder += 1

    return case
