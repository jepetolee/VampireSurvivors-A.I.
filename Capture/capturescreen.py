import cv2 as cv
import numpy as np
from multiprocessing import Pool
import pyautogui

def finding_character(src):
    temp = cv.imread("./data/hero.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    temp2 = cv.imread("./data/hero2.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp2, cv.TM_SQDIFF)
    min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(result)
    if min_val < min_val2:
        min_loc_def = min_loc
        h, w = temp.shape
    else:
        min_loc_def = min_loc2
        h, w = temp2.shape

    return min_loc_def[0], min_loc_def[1], h, w


def finding_boxes(src):
    temp = cv.imread("./data/box.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    results = []
    h, w = temp.shape

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

            res = (min_loc[0], min_loc[1], h, w)
            results.append(res)
        else:
            break
    return results


def finding_signs(src):  # capturing the treasure box's sign

    temp = cv.imread("./data/sign.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    results = []
    h, w = temp.shape
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    while True:
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if min_val < 200000:
            sx, sy = min_loc
            for x in range(sx, sx + w):
                for y in range(sy, sy + h):
                    try:
                        result[y][x] = 999999  # -MAX
                    except IndexError:  # ignore out of bounds
                        pass

            res = (min_loc[0], min_loc[1], h, w)
            results.append(res)
        else:
            break
    return results


def finding_gems(src):
    temp1 = cv.imread("./data/gem/crystal.png")
    temp1 = cv.cvtColor(temp1, cv.COLOR_RGB2GRAY)
    h1, w1 = temp1.shape
    temp1 = (temp1, h1, w1)
    temp2 = cv.imread("./data/gem/ruby.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)
    h2, w2 = temp2.shape
    temp2 = (temp2, h2, w2)
    temp3 = cv.imread("./data/gem/emerald.png")
    temp3 = cv.cvtColor(temp3, cv.COLOR_RGB2GRAY)
    h3, w3 = temp3.shape
    temp3 = (temp3, h3, w3)

    temp_all = [temp1, temp2, temp3]
    thresholds = [100000, 5000, 10000]
    result_bundle = []
    adder = 0
    for temp, h, w in temp_all:
        result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
        results = []
        while True:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            if min_val < thresholds[adder]:
                sx, sy = min_loc
                for x in range(sx - 10, sx + w + 10):
                    for y in range(sy - 10, sy + h + 10):
                        try:
                            result[y][x] = 999999  # -MAX
                            src[y][x] = 9999999
                        except IndexError:  # ignore out of bounds
                            pass

                res = (min_loc[0], min_loc[1], h, w)
                results.append(res)
            else:
                break
        adder = adder + 1
        result_bundle.append(results)

    return result_bundle


def finding_entities(src):
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp1 = cv.imread("./data/entity/big_bat.png")
    temp1 = cv.cvtColor(temp1, cv.COLOR_RGB2GRAY)
    h1, w1 = temp1.shape
    temp1 = (temp1, h1, w1)
    temp2 = cv.imread("./data/entity/golem_l.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)
    h2, w2 = temp2.shape
    temp2 = (temp2, h2, w2)
    temp3 = cv.imread("./data/entity/golem2_r.png")
    temp3 = cv.cvtColor(temp3, cv.COLOR_RGB2GRAY)
    h2, w2 = temp3.shape
    temp3 = (temp3, h2, w2)
    temp4 = cv.imread("./data/entity/bat_s_l.png")
    temp4 = cv.cvtColor(temp4, cv.COLOR_RGB2GRAY)
    h4, w4 = temp4.shape
    temp4 = (temp4, h4, w4)
    temp5 = cv.imread("./data/entity/bat_m_l.png")
    temp5 = cv.cvtColor(temp5, cv.COLOR_RGB2GRAY)
    h5, w5 = temp5.shape
    temp5 = (temp5, h5, w5)
    temp6 = cv.imread("./data/entity/bat_s_r.png")
    temp6 = cv.cvtColor(temp6, cv.COLOR_RGB2GRAY)
    h6, w6 = temp6.shape
    temp6 = (temp6, h6, w6)
    temp7 = cv.imread("./data/entity/bat_m_r.png")
    temp7 = cv.cvtColor(temp7, cv.COLOR_RGB2GRAY)
    h7, w7 = temp7.shape
    temp7 = (temp7, h7, w7)
    temp8 = cv.imread("./data/entity/golem_r.png")
    temp8 = cv.cvtColor(temp8, cv.COLOR_RGB2GRAY)
    h8, w8 = temp8.shape
    temp8 = (temp8, h8, w8)
    temp9 = cv.imread("./data/entity/golem2_l.png")
    temp9 = cv.cvtColor(temp9, cv.COLOR_RGB2GRAY)
    h9, w9 = temp9.shape
    temp9 = (temp9, h9, w9)
    temp10 = cv.imread("./data/entity/golem3_r.png")
    temp10 = cv.cvtColor(temp10, cv.COLOR_RGB2GRAY)
    h10, w10 = temp10.shape
    temp10 = (temp10, h10, w10)
    temp11 = cv.imread("./data/entity/golem3_l.png")
    temp11 = cv.cvtColor(temp11, cv.COLOR_RGB2GRAY)
    h11, w11 = temp11.shape
    temp11 = (temp11, h11, w11)
    temp12 = cv.imread("./data/entity/skeleton_l.png")
    temp12 = cv.cvtColor(temp12, cv.COLOR_RGB2GRAY)
    h12, w12 = temp12.shape
    temp12 = (temp12, h12, w12)
    temp13 = cv.imread("./data/entity/skeleton_r.png")
    temp13 = cv.cvtColor(temp13, cv.COLOR_RGB2GRAY)
    h13, w13 = temp13.shape
    temp13 = (temp13, h13, w13)
    temp14 = cv.imread("./data/entity/skeleton2_l.png")
    temp14 = cv.cvtColor(temp14, cv.COLOR_RGB2GRAY)
    h14, w14 = temp14.shape
    temp14 = (temp14, h14, w14)
    temp15 = cv.imread("./data/entity/skeleton2_r.png")
    temp15 = cv.cvtColor(temp15, cv.COLOR_RGB2GRAY)
    h15, w15 = temp15.shape
    temp15 = (temp15, h15, w15)
    temp16 = cv.imread("./data/entity/skeleton3_l.png")
    temp16 = cv.cvtColor(temp16, cv.COLOR_RGB2GRAY)
    h16, w16 = temp16.shape
    temp16 = (temp16, h16, w16)
    temp17 = cv.imread("./data/entity/skeleton3_r.png")
    temp17 = cv.cvtColor(temp17, cv.COLOR_RGB2GRAY)
    h17, w17 = temp17.shape
    temp17 = (temp17, h17, w17)
    temp18 = cv.imread("./data/entity/skeleton4_l.png")
    temp18 = cv.cvtColor(temp18, cv.COLOR_RGB2GRAY)
    h18, w18 = temp18.shape
    temp18 = (temp18, h18, w18)
    temp19 = cv.imread("./data/entity/skeleton4_r.png")
    temp19 = cv.cvtColor(temp19, cv.COLOR_RGB2GRAY)
    h19, w19 = temp19.shape
    temp19 = (temp19, h19, w19)
    temp20 = cv.imread("./data/entity/skeleton5_l.png")
    temp20 = cv.cvtColor(temp20, cv.COLOR_RGB2GRAY)
    h20, w20 = temp20.shape
    temp20 = (temp20, h20, w20)
    temp21 = cv.imread("./data/entity/skeleton5_r.png")
    temp21 = cv.cvtColor(temp21, cv.COLOR_RGB2GRAY)
    h21, w21 = temp21.shape
    temp21 = (temp21, h21, w21)
    temp22 = cv.imread("./data/entity/skeleton6_l.png")
    temp22 = cv.cvtColor(temp22, cv.COLOR_RGB2GRAY)
    h22, w22 = temp22.shape
    temp22 = (temp22, h22, w22)
    temp23 = cv.imread("./data/entity/skeleton6_r.png")
    temp23 = cv.cvtColor(temp23, cv.COLOR_RGB2GRAY)
    h23, w23 = temp23.shape
    temp23 = (temp23, h23, w23)
    temp24 = cv.imread("./data/entity/werewolf_l.png")
    temp24 = cv.cvtColor(temp24, cv.COLOR_RGB2GRAY)
    h24, w24 = temp24.shape
    temp24 = (temp24, h24, w24)
    temp25 = cv.imread("./data/entity/werewolf_r.png")
    temp25 = cv.cvtColor(temp25, cv.COLOR_RGB2GRAY)
    h25, w25 = temp25.shape
    temp25 = (temp25, h25, w25)
    temp26 = cv.imread("./data/entity/ghost_l.png")
    temp26 = cv.cvtColor(temp26, cv.COLOR_RGB2GRAY)
    h26, w26 = temp26.shape
    temp26 = (temp26, h26, w26)
    temp27 = cv.imread("./data/entity/ghost_r.png")
    temp27 = cv.cvtColor(temp27, cv.COLOR_RGB2GRAY)
    h27, w27 = temp27.shape
    temp27 = (temp27, h27, w27)

    temp_all = [temp1, temp2, temp3, temp4, temp5,
                temp6, temp7, temp8, temp9, temp10,
                temp11,temp12,temp13,temp14,temp15,
                temp16,temp17,temp18,temp19,temp20,
                temp21,temp22,temp23,temp24,temp25,
                temp26,temp27]

    threshold = [1000000, 100000, 900000, 500000, 500000,
                 700000, 1200000, 600000, 600000, 400000,
                 900000,6200000,5000000,6000000,5000000,
                 5500000,5000000,3000000,3500000,2500000,
                 3000000,2500000,3000000, 600000, 600000,
                 500000, 300000]
    result_bundle = []
    cnt = 0
    for temp, h, w in temp_all:
        result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
        results = []
        while True:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            print(min_val)
            if min_val < threshold[cnt]:

                sx, sy = min_loc
                for x in range(sx, sx + w):
                    for y in range(sy, sy + h):
                        try:
                            result[y][x] = 9999999
                            src[y][x] = 9999999 # -MAX
                        except IndexError:  # ignore out of bounds
                            pass

                res = (min_loc[0], min_loc[1], h, w)
                results.append(res)
            else:
                break
        cnt = cnt + 1
        print("cut")
        result_bundle.append(results)

    return result_bundle


def finding_boss_entities(src):
    temp1 = cv.imread("./data/boss_entity/boss.png")
    temp1 = cv.cvtColor(temp1, cv.COLOR_RGB2GRAY)
    h1, w1 = temp1.shape
    temp1 = (temp1, h1, w1)
    temp2 = cv.imread("./data/boss_entity/boss.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)
    h2, w2 = temp2.shape
    temp2 = (temp2, h2, w2)

    temp_all = [temp1, temp2]

    result_bundle = []
    for temp, h, w in temp_all:
        result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
        results = []
        while True:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            if min_val < 200000:
                sx, sy = min_loc
                for x in range(sx, sx + w):
                    for y in range(sy, sy + h):
                        try:
                            result[y][x] = 999999  # -MAX
                        except IndexError:  # ignore out of bounds
                            pass

                res = (min_loc[0], min_loc[1], h, w)
                results.append(res)
            else:

                break
        result_bundle.append(results)

    return result_bundle


# need to get multiprocessing&monitor capture to build data
def capture_screen():
    src = pyautogui.screenshot()
    img_frame = np.array(src)
    img_frame = cv.cvtColor(img_frame, cv.COLOR_RGB2GRAY)
    return img_frame


def main():
    img = cv.imread("test1.png")

    results = finding_entities(img)
    rand = 0
    green = 0
    blue = 0
    i = 0
    for result in results:
        for (x, y, h, w) in result:
            cv.rectangle(img, (x, y), (x + w, y + h), (blue, green, rand), 3)
        if i % 3 == 0:
            rand = rand + 30
        elif i % 3 == 1:
            green = green + 50
        else:
            blue = blue + 70
        i = i + 1

    cv.imwrite("output.png", img)


if __name__ == "__main__":
    main()
