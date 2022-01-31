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

def finding_box(src):
    src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    temp = cv.imread("./data/box.png")
    temp = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    result = cv.matchTemplate(src, temp, cv.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    h, w = temp.shape
    return (min_loc), h, w


def main():
    img = cv.imread("test1.png")

    pt,h,w = finding_character(img)

    cv.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
    cv.imwrite("output.png", img)


if __name__ == "__main__":
    main()




