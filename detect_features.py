import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import nonmaxsuppts


def detect_features(image):
    imgNew = image.copy()
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_RGB2GRAY)
    row, col = imgNew.shape
    imgNew = cv2.GaussianBlur(imgNew, (5,5), 0)

    sobelCol = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelRow = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    gradMatCol = cv2.filter2D(src = imgNew, ddepth = cv2.CV_64F, kernel = sobelCol)
    gradMatRow = cv2.filter2D(src = imgNew, ddepth = cv2.CV_64F, kernel = sobelRow)

    Ixx = gradMatCol**2
    Iyy = gradMatRow**2
    Ixy = gradMatCol*gradMatRow


    cornernessMat = np.zeros((row, col), np.float64)
    for y in range(1, row - 1):
        for x in range(1, col - 1):
            windowIxx = Ixx[y - 1:y + 1 + 1, x - 1:x + 1 + 1]
            windowIxy = Ixy[y - 1:y + 1 + 1, x - 1:x + 1 + 1]
            windowIyy = Iyy[y - 1:y + 1 + 1, x - 1:x + 1 + 1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            cornernessMat[y,x] = det - 0.04 * (trace ** 2)

    temp = np.amax(cornernessMat)
    pixel_coords = nonmaxsuppts.nonmaxsuppts(cornernessMat,10,temp*0.1) #0.06
    print len(pixel_coords)


    return pixel_coords

# pic = cv2.imread('bikes1.png')
# out = detect_features(pic)
#
# for ls in out:
#     cv2.circle(img=pic, center=(ls[1], ls[0]), radius=5, color=(0, 0, 255))
#     # image.itemset((ls[0], ls[1], 0), 0)
#     # image.itemset((ls[0], ls[1], 1), 0)
#     # image.itemset((ls[0], ls[1], 2), 255)
# cv2.imshow('image', pic)
# cv2.waitKey(0)