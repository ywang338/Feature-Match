import cv2
import numpy as np
import matplotlib.pyplot as plt
import detect_features
import match_features

def compute_affine_xform(matches,features1,features2,image1,image2):
    affine_xform = np.zeros((3,3))

    threshold = 1

    best_indice = []

    max_count = 0

    itreation = 100

    N = int(len(matches)/3)
    for i in range(1000):

        temp1 = np.random.randint(len(matches))
        temp2 = np.random.randint(len(matches))
        temp3 = np.random.randint(len(matches))

        m1 = matches[temp1]
        m2 = matches[temp2]
        m3 = matches[temp3]

        A = np.zeros([6, 6], int)
        B = np.zeros([6, 1], int)

        A = [[features1[m1[0]][0],features1[m1[0]][1], 1, 0, 0, 0],
             [0, 0, 0, features1[m1[0]][0], features1[m1[0]][1], 1],
             [features1[m2[0]][0], features1[m2[0]][1], 1, 0, 0, 0],
             [0, 0, 0, features1[m2[0]][0], features1[m2[0]][1], 1],
             [features1[m3[0]][0], features1[m3[0]][1], 1, 0, 0, 0],
             [0, 0, 0, features1[m3[0]][0], features1[m3[0]][1], 1],]

        B = [[features2[m1[1]][0]],
             [features2[m1[1]][1]],
             [features2[m2[1]][0]],
             [features2[m2[1]][1]],
             [features2[m3[1]][0]],
             [features2[m3[1]][1]]]


        # print A,B

        if np.linalg.det(A) == 0:
            continue

        T = np.dot(np.linalg.inv(A), B)

        # print T

        xfrm = np.zeros((3, 3), float)
        k = 0
        for i in range(2):
            for j in range(3):
                xfrm[i][j] = T[k][0]
                k += 1
        xfrm[2, 2] = 1

        # print xfrm


        ############ Calculate other features
        count = 0
        inliers_indice = []
        for i in matches:

            p1 = np.zeros((3,1),float)
            p1 = features1[i[0]]
            p1 = np.append(p1,1)
            p1 = np.transpose(p1)
            # print p1

            p2 = np.zeros((3, 1), float)
            p2 = features2[i[1]]
            p2 = np.append(p2, 1)
            p2 = np.transpose(p2)

            result_temp = np.dot(xfrm,p1)

            x_dis = p2[0] - result_temp[0]
            y_dis = p2[1] - result_temp[1]

            dis = x_dis**2+y_dis**2

            if dis <= threshold:
                count += 1
                inliers_indice.append([features1[i[0]], features2[i[1]]])


        if max_count <= count:
            max_count = count
            affine_xform = xfrm
            indice = inliers_indice

    return affine_xform,indice

