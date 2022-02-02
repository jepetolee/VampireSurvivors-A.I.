import cv2 as cv
import numpy as np
from multiprocessing import Pool


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
    temp2 = cv.imread("./data/entity/golem.png")
    temp2 = cv.cvtColor(temp2, cv.COLOR_RGB2GRAY)
    h2, w2 = temp2.shape
    temp2 = (temp2, h2, w2)
    temp3 = cv.imread("./data/entity/golem2.png")
    temp3 = cv.cvtColor(temp3, cv.COLOR_RGB2GRAY)
    h2, w2 = temp3.shape
    temp3 = (temp3, h2, w2)
    temp4 = cv.imread("./data/entity/bat.png")
    temp4 = cv.cvtColor(temp4, cv.COLOR_RGB2GRAY)
    h4, w4 = temp4.shape
    temp4 = (temp4, h4, w4)
    temp_all = [temp1, temp2, temp3, temp4]
    threshold = [1100000, 450000, 10000, 15000]
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
                            result[y][x] = 9999999  # -MAX
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


def main():
    img = cv.imread("test1.png")

    results = finding_entities(img)
    rand = 200
    green = 0

    for result in results:
        for (x, y, h, w) in result:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, green, rand), 3)
        rand = rand - 200
        green = 300
    cv.imwrite("output.png", img)


if __name__ == "__main__":
    main()
