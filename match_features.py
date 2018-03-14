import cv2
import numpy as np
import matplotlib.pyplot as plt
import detect_features


def match_features(feature_coords1, feature_coords2, image1, image2):

    window_size = 10
    dict_feature1 = {}
    dict_feature2 = {}

    for i in feature_coords1:
        a = i[0] - window_size
        b = i[1] - window_size
        c = i[0] + window_size
        d = i[1] + window_size
        temp = image1[a:c, b:d]
        if (temp.shape != (window_size * 2, window_size * 2)):
            continue
        max_p = -1
        index = [0,0]

        for j in feature_coords2:
            a1 = j[0] - window_size
            b1 = j[1] - window_size
            c1 = j[0] + window_size
            d1 = j[1] + window_size
            temp1 = image2[a1:c1, b1:d1]

            if (temp1.shape != (window_size * 2, window_size * 2)):
                continue
            product = np.mean((temp - temp.mean()) * (temp1 - temp1.mean()))
            stds = temp.std() * temp1.std()
            product /= stds
            if max_p < product:
                max_p = product
                index = j

        dict_feature1[i] = index


    for i in feature_coords2:

        a = i[0] - window_size
        b = i[1] - window_size
        c = i[0] + window_size
        d = i[1] + window_size
        temp = image2[a:c, b:d]
        if (temp.shape != (window_size * 2, window_size * 2)):
            continue
        max_p = -1
        index = [0,0]
        for j in feature_coords1:
            a1 = j[0] - window_size
            b1 = j[1] - window_size
            c1 = j[0] + window_size
            d1 = j[1] + window_size
            temp1 = image1[a1:c1, b1:d1]
            if (temp1.shape != (window_size * 2, window_size * 2)):
                continue
            product = np.mean((temp - temp.mean()) * (temp1 - temp1.mean()))
            stds = temp.std() * temp1.std()
            product /= stds
            if max_p < product:
                max_p = product
                index = j
        dict_feature2[i] = index

    print dict_feature1
    print dict_feature2

    matches = list()
    arr = list()
    for i in dict_feature1.keys():
        temp = dict_feature1[i]
        if dict_feature2[temp] == i:

            pos1 = feature_coords1.index(i)
            pos2 = feature_coords2.index(temp)
            arr.append([pos1, pos2])
            matches.append(i)
            matches.append(temp)


    return matches,arr

