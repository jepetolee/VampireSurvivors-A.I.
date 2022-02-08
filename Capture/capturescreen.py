import time

import cv2 as cv
import numpy as np
import pyautogui
import multiprocessing as mp
import pydirectinput
'''
def finding_boxes(src):
    temp = cv.imread("Capture/data/box.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    h, w = temp.shape
    image = np.zeros((1080, 1724), dtype=np.float32)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    while True:
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if min_val < 1760000:
            sx, sy = min_loc
            for x in range(sx, sx + w):
                for y in range(sy, sy + h):
                    try:
                        result[y][x] = 99999999  # -MAX
                    except IndexError:  # ignore out of bounds
                        pass

            cv.rectangle(image, (min_loc[0] - 100, min_loc[1], min_loc[0] - 100 + h, min_loc[1] + w), 150, -1)
        else:
            break

    return image


def finding_gems(src):
    temp1 = cv.imread("Capture/data/gem/crystal.png")
    temp1 = cv.cvtColor(temp1, cv.COLOR_RGB2GRAY)
    h1, w1 = temp1.shape
    temp1 = (temp1, h1, w1)
    temp2 = cv.imread("Capture/data/gem/ruby.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)
    h2, w2 = temp2.shape
    temp2 = (temp2, h2, w2)
    temp3 = cv.imread("Capture/data/gem/emerald.png")
    temp3 = cv.cvtColor(temp3, cv.COLOR_RGB2GRAY)
    h3, w3 = temp3.shape
    temp3 = (temp3, h3, w3)

    temp_all = [temp1, temp2, temp3]
    thresholds = [100000, 210000, 100000]
    image = np.zeros((1080, 1724), dtype=np.float32)
    adder = 0
    for temp, h, w in temp_all:
        result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
        while True:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            if min_val < thresholds[adder]:
                sx, sy = min_loc
                for x in range(sx - 10, sx + w + 10):
                    for y in range(sy - 10, sy + h + 10):
                        try:
                            result[y][x] = 999999  # -MAX
                        except IndexError:  # ignore out of bounds
                            pass

                cv.rectangle(image, (min_loc[0] - 100, min_loc[1], min_loc[0] + h - 100, min_loc[1] + w), 90, -1)
            else:
                break
        adder = adder + 1
    return image


def finding_entities(src):
    temp1 = cv.imread("Capture/data/entity/big_bat.png")
    temp1 = cv.cvtColor(temp1, cv.COLOR_RGB2GRAY)
    h1, w1 = temp1.shape
    temp1 = (temp1, h1, w1)
    temp2 = cv.imread("Capture/data/entity/golem_l.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)
    h2, w2 = temp2.shape
    temp2 = (temp2, h2, w2)
    temp3 = cv.imread("Capture/data/entity/golem2_r.png")
    temp3 = cv.cvtColor(temp3, cv.COLOR_RGB2GRAY)
    h2, w2 = temp3.shape
    temp3 = (temp3, h2, w2)
    temp4 = cv.imread("Capture/data/entity/bat_s_l.png")
    temp4 = cv.cvtColor(temp4, cv.COLOR_RGB2GRAY)
    h4, w4 = temp4.shape
    temp4 = (temp4, h4, w4)
    temp5 = cv.imread("Capture/data/entity/bat_m_l.png")
    temp5 = cv.cvtColor(temp5, cv.COLOR_RGB2GRAY)
    h5, w5 = temp5.shape
    temp5 = (temp5, h5, w5)
    temp6 = cv.imread("Capture/data/entity/bat_s_r.png")
    temp6 = cv.cvtColor(temp6, cv.COLOR_RGB2GRAY)
    h6, w6 = temp6.shape
    temp6 = (temp6, h6, w6)
    temp7 = cv.imread("Capture/data/entity/bat_m_r.png")
    temp7 = cv.cvtColor(temp7, cv.COLOR_RGB2GRAY)
    h7, w7 = temp7.shape
    temp7 = (temp7, h7, w7)
    temp8 = cv.imread("Capture/data/entity/golem_r.png")
    temp8 = cv.cvtColor(temp8, cv.COLOR_RGB2GRAY)
    h8, w8 = temp8.shape
    temp8 = (temp8, h8, w8)
    temp9 = cv.imread("Capture/data/entity/golem2_l.png")
    temp9 = cv.cvtColor(temp9, cv.COLOR_RGB2GRAY)
    h9, w9 = temp9.shape
    temp9 = (temp9, h9, w9)
    temp10 = cv.imread("Capture/data/entity/golem3_r.png")
    temp10 = cv.cvtColor(temp10, cv.COLOR_RGB2GRAY)
    h10, w10 = temp10.shape
    temp10 = (temp10, h10, w10)
    temp11 = cv.imread("Capture/data/entity/golem3_l.png")
    temp11 = cv.cvtColor(temp11, cv.COLOR_RGB2GRAY)
    h11, w11 = temp11.shape
    temp11 = (temp11, h11, w11)
    temp12 = cv.imread("Capture/data/entity/skeleton_l.png")
    temp12 = cv.cvtColor(temp12, cv.COLOR_RGB2GRAY)
    h12, w12 = temp12.shape
    temp12 = (temp12, h12, w12)
    temp13 = cv.imread("Capture/data/entity/skeleton_r.png")
    temp13 = cv.cvtColor(temp13, cv.COLOR_RGB2GRAY)
    h13, w13 = temp13.shape
    temp13 = (temp13, h13, w13)
    temp14 = cv.imread("Capture/data/entity/skeleton2_l.png")
    temp14 = cv.cvtColor(temp14, cv.COLOR_RGB2GRAY)
    h14, w14 = temp14.shape
    temp14 = (temp14, h14, w14)
    temp15 = cv.imread("Capture/data/entity/skeleton2_r.png")
    temp15 = cv.cvtColor(temp15, cv.COLOR_RGB2GRAY)
    h15, w15 = temp15.shape
    temp15 = (temp15, h15, w15)
    temp16 = cv.imread("Capture/data/entity/skeleton3_l.png")
    temp16 = cv.cvtColor(temp16, cv.COLOR_RGB2GRAY)
    h16, w16 = temp16.shape
    temp16 = (temp16, h16, w16)
    temp17 = cv.imread("Capture/data/entity/skeleton3_r.png")
    temp17 = cv.cvtColor(temp17, cv.COLOR_RGB2GRAY)
    h17, w17 = temp17.shape
    temp17 = (temp17, h17, w17)
    temp18 = cv.imread("Capture/data/entity/skeleton4_l.png")
    temp18 = cv.cvtColor(temp18, cv.COLOR_RGB2GRAY)
    h18, w18 = temp18.shape
    temp18 = (temp18, h18, w18)
    temp19 = cv.imread("Capture/data/entity/skeleton4_r.png")
    temp19 = cv.cvtColor(temp19, cv.COLOR_RGB2GRAY)
    h19, w19 = temp19.shape
    temp19 = (temp19, h19, w19)
    temp20 = cv.imread("Capture/data/entity/skeleton5_l.png")
    temp20 = cv.cvtColor(temp20, cv.COLOR_RGB2GRAY)
    h20, w20 = temp20.shape
    temp20 = (temp20, h20, w20)
    temp21 = cv.imread("Capture/data/entity/skeleton5_r.png")
    temp21 = cv.cvtColor(temp21, cv.COLOR_RGB2GRAY)
    h21, w21 = temp21.shape
    temp21 = (temp21, h21, w21)
    temp22 = cv.imread("Capture/data/entity/skeleton6_l.png")
    temp22 = cv.cvtColor(temp22, cv.COLOR_RGB2GRAY)
    h22, w22 = temp22.shape
    temp22 = (temp22, h22, w22)
    temp23 = cv.imread("Capture/data/entity/skeleton6_r.png")
    temp23 = cv.cvtColor(temp23, cv.COLOR_RGB2GRAY)
    h23, w23 = temp23.shape
    temp23 = (temp23, h23, w23)
    temp24 = cv.imread("Capture/data/entity/werewolf_l.png")
    temp24 = cv.cvtColor(temp24, cv.COLOR_RGB2GRAY)
    h24, w24 = temp24.shape
    temp24 = (temp24, h24, w24)
    temp25 = cv.imread("Capture/data/entity/werewolf_r.png")
    temp25 = cv.cvtColor(temp25, cv.COLOR_RGB2GRAY)
    h25, w25 = temp25.shape
    temp25 = (temp25, h25, w25)
    temp26 = cv.imread("Capture/data/entity/ghost_l.png")
    temp26 = cv.cvtColor(temp26, cv.COLOR_RGB2GRAY)
    h26, w26 = temp26.shape
    temp26 = (temp26, h26, w26)
    temp27 = cv.imread("Capture/data/entity/ghost_r.png")
    temp27 = cv.cvtColor(temp27, cv.COLOR_RGB2GRAY)
    h27, w27 = temp27.shape
    temp27 = (temp27, h27, w27)
    temp28 = cv.imread("Capture/data/entity/plant_l.png")
    temp28 = cv.cvtColor(temp28, cv.COLOR_RGB2GRAY)
    h28, w28 = temp28.shape
    temp28 = (temp28, h28, w28)
    temp29 = cv.imread("Capture/data/entity/plant_r.png")
    temp29 = cv.cvtColor(temp29, cv.COLOR_RGB2GRAY)
    h29, w29 = temp29.shape
    temp29 = (temp29, h29, w29)
    temp30 = cv.imread("Capture/data/entity/neplant.png")
    temp30 = cv.cvtColor(temp30, cv.COLOR_RGB2GRAY)
    h30, w30 = temp30.shape
    temp30 = (temp30, h30, w30)
    temp31 = cv.imread("Capture/data/entity/neplant2.png")
    temp31 = cv.cvtColor(temp31, cv.COLOR_RGB2GRAY)
    h31, w31 = temp31.shape
    temp31 = (temp31, h31, w31)
    temp32 = cv.imread("Capture/data/entity/neplant3.png")
    temp32 = cv.cvtColor(temp32, cv.COLOR_RGB2GRAY)
    h32, w32 = temp32.shape
    temp32 = (temp32, h32, w32)
    temp33 = cv.imread("Capture/data/entity/neplant4.png")
    temp33 = cv.cvtColor(temp33, cv.COLOR_RGB2GRAY)
    h33, w33 = temp33.shape
    temp33 = (temp33, h33, w33)
    temp34 = cv.imread("Capture/data/entity/mummy_l.png")
    temp34 = cv.cvtColor(temp34, cv.COLOR_RGB2GRAY)
    h34, w34 = temp34.shape
    temp34 = (temp34, h34, w34)
    temp35 = cv.imread("Capture/data/entity/mummy_r.png")
    temp35 = cv.cvtColor(temp35, cv.COLOR_RGB2GRAY)
    h35, w35 = temp35.shape
    temp35 = (temp35, h35, w35)
    temp36 = cv.imread("Capture/data/entity/zombie_l.png")
    temp36 = cv.cvtColor(temp36, cv.COLOR_RGB2GRAY)
    h36, w36 = temp36.shape
    temp36 = (temp36, h36, w36)
    temp37 = cv.imread("Capture/data/entity/zombie_r.png")
    temp37 = cv.cvtColor(temp37, cv.COLOR_RGB2GRAY)
    h37, w37 = temp37.shape
    temp37 = (temp37, h37, w37)
    temp38 = cv.imread("Capture/data/entity/huge_plant_l.png")
    temp38 = cv.cvtColor(temp38, cv.COLOR_RGB2GRAY)
    h38, w38 = temp38.shape
    temp38 = (temp38, h38, w38)
    temp39 = cv.imread("Capture/data/entity/huge_plant_r.png")
    temp39 = cv.cvtColor(temp39, cv.COLOR_RGB2GRAY)
    h39, w39 = temp39.shape
    temp39 = (temp39, h39, w39)

    temp_all = [temp1, temp2, temp3, temp4, temp5,
                temp6, temp7, temp8, temp9, temp10,
                temp11, temp12, temp13, temp14, temp15,
                temp16, temp17, temp18, temp19, temp20,
                temp21, temp22, temp23, temp24, temp25,
                temp26, temp27, temp28, temp29, temp30,
                temp31, temp32, temp33, temp34, temp35,
                temp36, temp37, temp38, temp39]

    threshold = [1000000, 100000, 900000, 500000, 500000,
                 700000, 100000, 600000, 600000, 400000,
                 900000, 6200000, 4500000, 6000000, 5000000,
                 5500000, 5000000, 3000000, 3500000, 2700000,
                 3000000, 2500000, 3000000, 600000, 600000,
                 470000, 200000, 4000000, 6000000, 3000000,
                 5500000, 4700000, 3500000, 2300000, 1200000,
                 320000, 300000, 550000, 200000]
    image = np.zeros((1080, 1724), dtype=np.float32)
    cnt = 0
    for temp, h, w in temp_all:
        result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
        while True:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            if min_val < threshold[cnt]:
                sx, sy = min_loc
                for x in range(sx, sx + w):
                    for y in range(sy, sy + h):
                        try:
                            result[y][x] = 9999999  # -MAX
                        except IndexError:  # ignore out of bounds
                            pass

                cv.rectangle(image, (min_loc[0] - 100, min_loc[1], min_loc[0] + h - 100, min_loc[1] + w), 60, -1)
            else:
                break
        cnt = cnt + 1

    return image
    # tcl


def finding_boss_entities(src):
    temp1 = cv.imread("Capture/data/boss_entity/boss_bat_l.png")
    temp1 = cv.cvtColor(temp1, cv.COLOR_RGB2GRAY)
    h1, w1 = temp1.shape
    temp1 = (temp1, h1, w1)
    temp2 = cv.imread("Capture/data/boss_entity/boss_bat_r.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)
    h2, w2 = temp2.shape
    temp2 = (temp2, h2, w2)
    temp3 = cv.imread("Capture/data/boss_entity/boss_bat2_r.png")
    temp3 = cv.cvtColor(temp3, cv.COLOR_RGB2GRAY)
    h3, w3 = temp3.shape
    temp3 = (temp3, h3, w3)
    temp4 = cv.imread("Capture/data/boss_entity/boss_bat2_l.png")
    temp4 = cv.cvtColor(temp4, cv.COLOR_RGB2GRAY)
    h4, w4 = temp4.shape
    temp4 = (temp4, h4, w4)
    temp5 = cv.imread("Capture/data/boss_entity/mantis_s_l.png")
    temp5 = cv.cvtColor(temp5, cv.COLOR_RGB2GRAY)
    h5, w5 = temp5.shape
    temp5 = (temp5, h5, w5)
    temp6 = cv.imread("Capture/data/boss_entity/mantis_s_r.png")
    temp6 = cv.cvtColor(temp6, cv.COLOR_RGB2GRAY)
    h6, w6 = temp6.shape
    temp6 = (temp6, h6, w6)
    temp7 = cv.imread("Capture/data/boss_entity/mantis_b_l.png")
    temp7 = cv.cvtColor(temp7, cv.COLOR_RGB2GRAY)
    h7, w7 = temp7.shape
    temp7 = (temp7, h7, w7)
    temp8 = cv.imread("Capture/data/boss_entity/mantis_b_r.png")
    temp8 = cv.cvtColor(temp8, cv.COLOR_RGB2GRAY)
    h8, w8 = temp8.shape
    temp8 = (temp8, h8, w8)
    image = np.zeros((1080, 1724), dtype=np.float32)
    temp_all = [temp1, temp2, temp3, temp4, temp7, temp8, temp5, temp6]
    thresholds = [2500000, 1300000, 900000, 930000, 120025, 100000, 2000000, 2000000]
    adder = 0

    for temp, h, w in temp_all:
        result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
        while True:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            if min_val < thresholds[adder]:

                sx, sy = min_loc
                for x in range(sx - 10, sx + w + 10):
                    for y in range(sy - 10, sy + h + 10):
                        try:
                            result[y][x] = 99999999  # -MAX
                        except IndexError:  # ignore out of bounds
                            pass
                cv.rectangle(image, (min_loc[0] - 100, min_loc[1], min_loc[0] + h - 100, min_loc[1] + w), 30, -1)
            else:
                break
        adder = adder + 1

    return image
'''

def capture_screen():
    src = pyautogui.screenshot(region=(100, 0, 1724, 1080))
    img_frame = np.array(src)
    img_frame = cv.cvtColor(img_frame, cv.COLOR_RGB2GRAY)
    cv.rectangle(img_frame, (830, 460), (920, 560), 255, -1)
    # entity = finding_entities(img_frame)
    # boss_entity = finding_boss_entities(img_frame)
    # boxes = finding_boxes(img_frame)
    # gems = finding_gems(img_frame)
    '''
    funcs = [finding_entities,finding_gems,finding_boxes,finding_boss_entities]
    process = [mp.Process(target=funcs[i],args=(img_frame,))for i in range(len(funcs))]
    for proces in process:
        proces.start()'''

    # image = (img_frame, entity, boss_entity, boxes, gems)
    return img_frame


def item_selection():
    src = pyautogui.screenshot()
    src = np.array(src)
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    dropbox = cv.imread("Capture/data/dropbox.png")
    dropbox = cv.cvtColor(dropbox, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, dropbox, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    select = cv.imread("Capture/data/selection.png")
    select = cv.cvtColor(select, cv.COLOR_RGB2GRAY)
    result2 = cv.matchTemplate(src, select, cv.TM_SQDIFF)
    min_val1, max_val2, min_loc1, max_loc2 = cv.minMaxLoc(result2)
    if min_val < 15000000:
        pyautogui.moveTo(950, 880)
        pyautogui.click()
        time.sleep(20)
        pyautogui.moveTo(950, 880)
        pyautogui.click()
        return 1

    elif min_val1 < 1468192:

        pyautogui.moveTo(930, 540)
        pyautogui.click()
        return 2
    return 0


def selection():
    src = pyautogui.screenshot()
    src = np.array(src)
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp = cv.imread("Capture/data/selection.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    if min_val < 14000000:
        return True
    else:
        return False


def game_over():
    src = pyautogui.screenshot()
    src = np.array(src)
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp = cv.imread("Capture/data/end.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    if min_val < 14000000:
        return True
    else:
        return False
