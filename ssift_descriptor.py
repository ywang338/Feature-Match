import random
import cv2
import numpy as np
import sys
import detect_features
import match_features
import compute_affine_xform
import compute_proj_xform

def ssift_descriptor(feature_coords,image):

    descriptors = dict()
    eightElementList = []

    sobelN = np.array([[1,2,1], [0,0,0], [-1,2,-1]])
    sobelNE = np.array([[0,1,2], [-1,0,-1], [2, -1, 0]])
    sobelE = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelES = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    sobelS = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobelSW = np.array([[0, -1, 2], [1, 0, -1], [2, 1, 0]])
    sobelW = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelWN = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    gradMat = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=sobelN)
    eightElementList.append(gradMat)
    gradMat = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=sobelNE)
    eightElementList.append(gradMat)
    gradMat = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=sobelE)
    eightElementList.append(gradMat)
    gradMat = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=sobelES)
    eightElementList.append(gradMat)
    gradMat = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=sobelS)
    eightElementList.append(gradMat)
    gradMat = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=sobelSW)
    eightElementList.append(gradMat)
    gradMat = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=sobelW)
    eightElementList.append(gradMat)
    gradMat = cv2.filter2D(src=image, ddepth=cv2.CV_64F, kernel=sobelWN)
    eightElementList.append(gradMat)


    for coord in feature_coords:
        windowSize = 20
        row, col = image.shape
        if (coord[0] + windowSize >= row) or (coord[0] - windowSize <= 0) or \
                         (coord[1]+windowSize>=col)or(coord[1]-windowSize<=0):continue
        y = coord[0] - 20
        x = coord[1] - 20
        window = np.zeros((41, 41), np.matrix)
        for i in range(41):
            for j in range(41):
                window[i,j] = y,x
                x+=1
            y+=1
            x=coord[1] - 20

        size = 10
        vec = np.zeros((4,4,8))
        for m in range(4):
            for n in range(4):
                grid = window[m*size:m*size+size,n*size:n*size+size]
                cnt = 0
                for i in eightElementList:
                    sum = 0
                    for j in grid:
                        for k in range(10):
                            sum+= i[j[k]]
                    vec[m,n,cnt] = sum
                    cnt += 1

        vec = np.resize(vec,(128,1))
        vec= vec/np.dot(vec.T,vec)
        for i in vec:
            if i > 0.2: i = 0.2

        vec = vec / np.dot(vec.T, vec)
        descriptors[coord] = vec
    return descriptors


def ratio_test (descriptor1 , descriptor2):
    a = descriptor1
    b = descriptor2
    dic = {}
    dic2Min = {}
    for i in a:
        min = secondMin= sys.maxint
        dic[i] = 0,0
        for j in b:
            dist = ((a[i]-b[j])**2).sum()
            if(dist< min):
                secondMin = min
                dic2Min[i] = dic[i]
                min = dist
                dic[i] = j
            elif ((dist<secondMin) and (dist!=min)):
                    secondMin = dist
                    dic2Min[i] = j
    for i in dic:
        dist1 = ((a[i] - b[dic[i]]) ** 2).sum()
        dist2 = ((a[i] - b[dic2Min[i]]) ** 2).sum()
        if(dist1/dist2)>0.6:
            dic[i] = 0

    return dic


