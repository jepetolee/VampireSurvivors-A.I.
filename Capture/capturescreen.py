import cv2 as cv
import numpy as np
import time


def finding_character(src):
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp = cv.imread("./data/hero.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    h, w = temp.shape

    return (min_loc), h, w

def finding_boxes(src):
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp = cv.imread("./data/box.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    points = []
    h, w = temp.shape
    while True:
         result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
         min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

         if min_val<200000:
             sx, sy = min_loc
             for x in range(sx, sx+w):
                 for y in range(sy , sy + h):
                     try:
                         src[y][x] = np.float32(-10000)  # -MAX
                     except IndexError:  # ignore out of bounds
                         pass
             point = (min_loc[0],min_loc[1],h,w)
             points.append(point)
         else:
             break
    return points


def finding_signs(src):#capturing the treasure box's sign
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp = cv.imread("./data/sign.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    points = []
    h, w = temp.shape
    while True:
        result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if min_val < 200000:
            sx, sy = min_loc
            for x in range(sx, sx + w):
                for y in range(sy, sy + h):
                    try:
                        src[y][x] = np.float32(-10000)  # -MAX
                    except IndexError:  # ignore out of bounds
                        pass
            point = (min_loc[0], min_loc[1], h, w)
            points.append(point)
        else:
            break
    return points

def finding_gems(src):
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp = cv.imread("./data/hero.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    h, w = temp.shape
    return (min_loc), h, w

def finding_entities(src):
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

    temp = cv.imread("./data/entity/entity.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    h, w = temp.shape
    return (min_loc), h, w

def finding_boss_entities(src):
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp = cv.imread("./data/hero.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    h, w = temp.shape
    return (min_loc), h, w


def main():
    img = cv.imread("test1.png")

    result = finding_signs(img)

    for (x,y,h,w)  in result:
        cv.rectangle(img, (x, y), (x + w, y+ h), (0, 0, 200), 3)
    cv.imwrite("output.png", img)


if __name__ == "__main__":
    main()




