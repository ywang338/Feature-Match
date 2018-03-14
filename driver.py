import cv2
import numpy as np
import matplotlib.pyplot as plt
import detect_features
import match_features
import compute_affine_xform
import compute_fully_projective_xform
import ssift_descriptor

##############          Feature Detection               ##############

# window for bikes1,2 is 10 and the threshold is 0.1*max_corneress
# window for bikes1,3 is 15 and the threshold is 0.1*max_corneress
# window for leuven1,2 and 3 is 10 and the threshold is 0.1*max_corneress
# window for graf1,2 and 3 is 5 and the threshold is 0.6*max_corneress
# window for wall1,2,3 is 5 and the threshold is 0.6*max_corneress



pic = cv2.imread('bikes1.png')
out = detect_features.detect_features(pic)

pic1 = cv2.imread('bikes2.png')
out1 = detect_features.detect_features(pic1)

pic2 = cv2.imread('bikes3.png')
out2 = detect_features.detect_features(pic2)

for i in out:
    cv2.circle(pic,(i[1],i[0]),5,(0,255,0))

for i in out1:
    cv2.circle(pic1,(i[1],i[0]),5,(0,255,0))

for i in out1:
    cv2.circle(pic2,(i[1],i[0]),5,(0,255,0))


cv2.imwrite("Feature_bikes1.jpg",pic)
cv2.imwrite("Feature_bikes2.jpg",pic1)
cv2.imwrite("Feature_bikes3.jpg",pic2)


pic = cv2.imread('leuven1.png')
out = detect_features.detect_features(pic)

pic1 = cv2.imread('leuven2.png')
out1 = detect_features.detect_features(pic1)

pic2 = cv2.imread('leuven3.png')
out2 = detect_features.detect_features(pic2)

for i in out:
    cv2.circle(pic,(i[1],i[0]),5,(0,255,0))

for i in out1:
    cv2.circle(pic1,(i[1],i[0]),5,(0,255,0))

for i in out1:
    cv2.circle(pic2,(i[1],i[0]),5,(0,255,0))


cv2.imwrite("Feature_leuven1.jpg",pic)
cv2.imwrite("Feature_leuven2.jpg",pic1)
cv2.imwrite("Feature_leuven3.jpg",pic2)



pic = cv2.imread('graf1.png')
out = detect_features.detect_features(pic)

pic1 = cv2.imread('graf2.png')
out1 = detect_features.detect_features(pic1)

pic2 = cv2.imread('graf3.png')
out2 = detect_features.detect_features(pic2)

for i in out:
    cv2.circle(pic,(i[1],i[0]),5,(0,255,0))

for i in out1:
    cv2.circle(pic1,(i[1],i[0]),5,(0,255,0))

for i in out1:
    cv2.circle(pic2,(i[1],i[0]),5,(0,255,0))


cv2.imwrite("Feature_graf1.jpg",pic)
cv2.imwrite("Feature_graf2.jpg",pic1)
cv2.imwrite("Feature_graf3.jpg",pic2)


pic = cv2.imread('wall1.png')
out = detect_features.detect_features(pic)

pic1 = cv2.imread('wall2.png')
out1 = detect_features.detect_features(pic1)

pic2 = cv2.imread('wall3.png')
out2 = detect_features.detect_features(pic2)

for i in out:
    cv2.circle(pic,(i[1],i[0]),5,(0,255,0))

for i in out1:
    cv2.circle(pic1,(i[1],i[0]),5,(0,255,0))

for i in out1:
    cv2.circle(pic2,(i[1],i[0]),5,(0,255,0))


cv2.imwrite("Feature_wall1.jpg",pic)
cv2.imwrite("Feature_wall2.jpg",pic1)
cv2.imwrite("Feature_wall3.jpg",pic2)



#############              Matching                    ##############

############# Test for bikes1, bikes2 and bikes3 ################

# for bikes1 and bikes 2 the window in monmax is 10 threshold 0.1*max_cornerness
# the window in match is 10

# for bikes1 and bikes 3 the window in nonmax is 15 threshold 0.1*max_cornerness
# the window in match is 10



pic = cv2.imread('bikes1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('bikes2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

pic2 = cv2.imread('bikes3.png')
out2 = detect_features.detect_features(pic2)
gray2 = cv2.cvtColor(pic2, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)




newImage = np.concatenate((pic,pic1),axis= 1)
offset = pic1.shape[1]

for i in range(0,len(results)-1,2):
    cv2.line(newImage,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(newImage,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(newImage, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

#cv2.imwrite("Feature_matches_between_bikes12.jpg",newImage)


results,index = match_features.match_features(out, out2, gray, gray2)

image = np.concatenate((pic,pic2),axis= 1)
offset = pic2.shape[1]

for i in range(0,len(results)-1,2):
    cv2.line(image,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(image,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(image, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))


# cv2.imwrite("Feature_matches_between_bikes13.jpg", image)





############# Test for leuven1, leuven2 and leuven3 ################

# for leuven1 and leuven2 the window in monmax is 10 threshold 0.1*max_cornerness
# the window in match is 10

# for leuven1 and leuven3 the window in nonmax is 10 threshold 0.1*max_cornerness
# the window in match is 10

pic = cv2.imread('leuven1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('leuven2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

pic2 = cv2.imread('leuven3.png')
out2 = detect_features.detect_features(pic2)
gray2 = cv2.cvtColor(pic2, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)



newImage = np.concatenate((pic,pic1),axis= 1)
offset = pic1.shape[1]

for i in range(0,len(results)-1,2):
    cv2.line(newImage,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(newImage,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(newImage, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

# cv2.imwrite("Feature_matches_between_leuven12.jpg",newImage)


results,index = match_features.match_features(out, out2, gray, gray2)

image = np.concatenate((pic,pic2),axis= 1)
offset = pic2.shape[1]

for i in range(0,len(results)-1,2):
    cv2.line(image,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(image,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(image, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

# cv2.imwrite("Feature_matches_between_leuven13.jpg", image)





############# Test for graf1, graf2 and graf3 ################

# for graf1 and graf2 the window in monmax is 5 threshold 0.6*max_cornerness
# the window in match is 5

# for graf1 and graf3 the window in nonmax is 5 threshold 0.6*max_cornerness
# the window in match is 5

pic = cv2.imread('graf1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('graf2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

pic2 = cv2.imread('graf3.png')
out2 = detect_features.detect_features(pic2)
gray2 = cv2.cvtColor(pic2, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)



newImage = np.concatenate((pic,pic1),axis= 1)
offset = pic1.shape[1]

for i in range(0,len(results)-1,2):
    cv2.line(newImage,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(newImage,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(newImage, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

# cv2.imwrite("Feature_matches_between_graf12.jpg",newImage)


results,index = match_features.match_features(out, out2, gray, gray2)

image = np.concatenate((pic,pic2),axis= 1)
offset = pic2.shape[1]

for i in range(0,len(results)-1,2):
    cv2.line(image,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(image,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(image, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))


# cv2.imwrite("Feature_matches_between_graf13.jpg", image)



############ Test for wall1, wall2 and wall3 ################

# for wall1 and wall2 the window in monmax is 5 threshold 0.6*max_cornerness
# the window in match is 5
#
# for wall1 and wall3 the window in nonmax is 5 threshold 0.6*max_cornerness
# the window in match is 5

pic = cv2.imread('wall1.png')
pic_size_col =pic.shape[1]
pic_size_row =pic.shape[0]

out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1_temp = cv2.imread('wall2.png')
pic1 = cv2.resize(pic1_temp,(1000,700),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

pic2_temp = cv2.imread('wall3.png')
pic2 = cv2.resize(pic2_temp,(1000,700),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
out2 = detect_features.detect_features(pic2)
gray2 = cv2.cvtColor(pic2, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)


print pic.shape
print pic1.shape

newImage = np.concatenate((pic,pic1),axis= 1)
offset = pic1.shape[1]

for i in range(0,len(results)-1,2):
    cv2.line(newImage,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(newImage,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(newImage, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

# cv2.imwrite("Feature_matches_between_wall12.jpg",newImage)


results,index = match_features.match_features(out, out2, gray, gray2)

image = np.concatenate((pic,pic2),axis= 1)
offset = pic2.shape[1]

for i in range(0,len(results)-1,2):
    cv2.line(image,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(image,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(image, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

cv2.imwrite("Feature_matches_between_wall13.jpg", image)


#############          Alignment and  Stitching               ##############


pic = cv2.imread('bikes2.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('bikes1.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)

affine_xform, indice = compute_affine_xform.compute_affine_xform(index,out,out1,gray,gray1)

com_img = np.concatenate((pic, pic1), axis=1)

offset = pic1.shape[1]

for i in indice:
    pt1 = i[0]
    pt2 = i[1]
    cv2.line(com_img,(pt1[1],pt1[0]),(pt2[1]+offset,pt2[0]),(225,0,0),2)

for i in range(0,len(results)-1,2):
    cv2.line(com_img,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(com_img,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(com_img, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))


# cv2.imwrite("In_out_features_bikes12.jpg",com_img)

### show the results of affine

temp_affine = np.array([[affine_xform[0][0], affine_xform[0][1], affine_xform[0][2]],[affine_xform[1][0], affine_xform[1][1], affine_xform[1][2]]])


dst = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((dst, pic), axis=1)

# cv2.imwrite("allign_bikes12.jpg",final)

newImage = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic
cv2.imwrite("stiching_bikes12.jpg",newImage)





pic = cv2.imread('bikes3.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('bikes1.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)

affine_xform, indice = compute_affine_xform.compute_affine_xform(index,out,out1,gray,gray1)

com_img = np.concatenate((pic, pic1), axis=1)

offset = pic1.shape[1]

for i in indice:
    pt1 = i[0]
    pt2 = i[1]
    cv2.line(com_img,(pt1[1],pt1[0]),(pt2[1]+offset,pt2[0]),(225,0,0),2)

for i in range(0,len(results)-1,2):
    cv2.line(com_img,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(com_img,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(com_img, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))


cv2.imwrite("In_out_features_bikes13.jpg",com_img)

### show the results of affine

temp_affine = np.array([[affine_xform[0][0], affine_xform[0][1], affine_xform[0][2]],[affine_xform[1][0], affine_xform[1][1], affine_xform[1][2]]])

dst = cv2.warpAffine(pic, temp_affine, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((dst, pic), axis=1)

cv2.imwrite("allign_bikes13.jpg",final)

newImage = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic
cv2.imwrite("stiching_bikes13.jpg",newImage)





pic = cv2.imread('leuven2.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('leuven1.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)

affine_xform, indice = compute_affine_xform.compute_affine_xform(index,out,out1,gray,gray1)

com_img = np.concatenate((pic, pic1), axis=1)

offset = pic1.shape[1]

for i in indice:
    pt1 = i[0]
    pt2 = i[1]
    cv2.line(com_img,(pt1[1],pt1[0]),(pt2[1]+offset,pt2[0]),(225,0,0),2)

for i in range(0,len(results)-1,2):
    cv2.line(com_img,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(com_img,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(com_img, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

cv2.imwrite("In_out_features_leuven12.jpg",com_img)

### show the results of affine

temp_affine = np.array([[affine_xform[0][0], affine_xform[0][1], affine_xform[0][2]],[affine_xform[1][0], affine_xform[1][1], affine_xform[1][2]]])

dst = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((dst, pic), axis=1)

cv2.imwrite("allign_leuven12.jpg",final)

newImage = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic
cv2.imwrite("stiching_leuven12.jpg",newImage)






pic = cv2.imread('leuven3.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('leuven1.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)

affine_xform, indice = compute_affine_xform.compute_affine_xform(index,out,out1,gray,gray1)

com_img = np.concatenate((pic, pic1), axis=1)

offset = pic1.shape[1]

for i in indice:
    pt1 = i[0]
    pt2 = i[1]
    cv2.line(com_img,(pt1[1],pt1[0]),(pt2[1]+offset,pt2[0]),(225,0,0),2)

for i in range(0,len(results)-1,2):
    cv2.line(com_img,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(com_img,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(com_img, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))



cv2.imwrite("In_out_features_leuven13.jpg",com_img)

### show the results of affine

temp_affine = np.array([[affine_xform[0][0], affine_xform[0][1], affine_xform[0][2]],[affine_xform[1][0], affine_xform[1][1], affine_xform[1][2]]])

dst = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((dst, pic), axis=1)

cv2.imwrite("allign_leuven13.jpg",final)

newImage = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic
cv2.imwrite("stiching_leuven13.jpg",newImage)



pic = cv2.imread('graf1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('graf2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)

affine_xform, indice = compute_affine_xform.compute_affine_xform(index,out,out1,gray,gray1)

com_img = np.concatenate((pic, pic1), axis=1)

offset = pic1.shape[1]

for i in indice:
    pt1 = i[0]
    pt2 = i[1]
    cv2.line(com_img,(pt1[1],pt1[0]),(pt2[1]+offset,pt2[0]),(225,0,0),2)

for i in range(0,len(results)-1,2):
    cv2.line(com_img,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(com_img,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(com_img, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

cv2.imwrite("In_out_features_graf12.jpg",com_img)

### show the results of affine

temp_affine = np.array([[affine_xform[0][0], affine_xform[0][1], affine_xform[0][2]],[affine_xform[1][0], affine_xform[1][1], affine_xform[1][2]]])

dst = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((dst, pic), axis=1)

cv2.imwrite("allign_graf12.jpg",final)

newImage = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic
cv2.imwrite("stiching_graf12.jpg",newImage)




pic = cv2.imread('graf1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('graf3.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)

affine_xform, indice = compute_affine_xform.compute_affine_xform(index,out,out1,gray,gray1)

com_img = np.concatenate((pic, pic1), axis=1)

offset = pic1.shape[1]

for i in indice:
    pt1 = i[0]
    pt2 = i[1]
    cv2.line(com_img,(pt1[1],pt1[0]),(pt2[1]+offset,pt2[0]),(225,0,0),2)

for i in range(0,len(results)-1,2):
    cv2.line(com_img,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(com_img,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(com_img, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

cv2.imwrite("In_out_features_graf13.jpg",com_img)

### show the results of affine

temp_affine = np.array([[affine_xform[0][0], affine_xform[0][1], affine_xform[0][2]],[affine_xform[1][0], affine_xform[1][1], affine_xform[1][2]]])

dst = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((dst, pic), axis=1)

cv2.imwrite("allign_graf13.jpg",final)

newImage = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic
cv2.imwrite("stiching_graf13.jpg",newImage)





pic_temp = cv2.imread('wall2.png')
pic = cv2.resize(pic_temp, (1000, 700))

out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('wall1.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)

affine_xform, indice = compute_affine_xform.compute_affine_xform(index,out,out1,gray,gray1)

com_img = np.concatenate((pic, pic1), axis=1)

offset = pic1.shape[1]

for i in indice:
    pt1 = i[0]
    pt2 = i[1]
    cv2.line(com_img,(pt1[1],pt1[0]),(pt2[1]+offset,pt2[0]),(225,0,0),2)

for i in range(0,len(results)-1,2):
    cv2.line(com_img,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(com_img,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(com_img, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

cv2.imwrite("In_out_features_wall12.jpg",com_img)

### show the results of affine

temp_affine = np.array([[affine_xform[0][0], affine_xform[0][1], affine_xform[0][2]],[affine_xform[1][0], affine_xform[1][1], affine_xform[1][2]]])

dst = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((dst, pic), axis=1)

cv2.imwrite("allign_wall12.jpg",final)

newImage = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic
cv2.imwrite("stiching_wall12.jpg",newImage)




pic_temp = cv2.imread('wall3.png')
pic = cv2.resize(pic_temp, (1000, 700))

out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('wall1.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results,index = match_features.match_features(out, out1, gray, gray1)

affine_xform, indice = compute_affine_xform.compute_affine_xform(index,out,out1,gray,gray1)

com_img = np.concatenate((pic, pic1), axis=1)

offset = pic1.shape[1]

for i in indice:
    pt1 = i[0]
    pt2 = i[1]
    cv2.line(com_img,(pt1[1],pt1[0]),(pt2[1]+offset,pt2[0]),(225,0,0),2)

for i in range(0,len(results)-1,2):
    cv2.line(com_img,(results[i][1],results[i][0]),(results[i+1][1]+offset,results[i+1][0]),(0,0,255))
    cv2.circle(com_img,(results[i][1],results[i][0]),5,(0,255,0))
    cv2.circle(com_img, (results[i+1][1]+offset,results[i+1][0]), 5, (0, 255, 0))

cv2.imwrite("In_out_features_wall13.jpg",com_img)

### show the results of affine

temp_affine = np.array([[affine_xform[0][0], affine_xform[0][1], affine_xform[0][2]],[affine_xform[1][0], affine_xform[1][1], affine_xform[1][2]]])

dst = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((dst, pic), axis=1)

cv2.imwrite("allign_wall13.jpg",final)

newImage = cv2.warpAffine(pic1, temp_affine, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic
cv2.imwrite("stiching_wall13.jpg",newImage)



#############    fully_pro ###############


pic = cv2.imread('bikes1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('bikes2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results, index = match_features.match_features(out, out1, gray, gray1)

affine_xform = compute_fully_projective_xform.compute_proj_xform(index, out, out1, gray, gray1)

dst = cv2.warpPerspective(pic1, affine_xform, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((pic, dst), axis=1)
cv2.imwrite("alignment_bikes12_p.jpg",final)

newImage = cv2.warpPerspective(pic, affine_xform, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic1
cv2.imwrite("stitching_bikes12_p.jpg",newImage)




pic = cv2.imread('bikes1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('bikes3.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results, index = match_features.match_features(out, out1, gray, gray1)

affine_xform = compute_fully_projective_xform.compute_proj_xform(index, out, out1, gray, gray1)

dst = cv2.warpPerspective(pic1, affine_xform, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((pic, dst), axis=1)
cv2.imwrite("alignment_bikes13_p.jpg",final)

newImage = cv2.warpPerspective(pic, affine_xform, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic1
cv2.imwrite("stitching_bikes13_p.jpg",newImage)





pic = cv2.imread('leuven1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('leuven2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results, index = match_features.match_features(out, out1, gray, gray1)

affine_xform = compute_fully_projective_xform.compute_proj_xform(index, out, out1, gray, gray1)

dst = cv2.warpPerspective(pic1, affine_xform, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((pic, dst), axis=1)
cv2.imwrite("alignment_leuven12_p.jpg",final)

newImage = cv2.warpPerspective(pic, affine_xform, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic1
cv2.imwrite("stitching_leuven12_p.jpg",newImage)





pic = cv2.imread('leuven1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('leuven3.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results, index = match_features.match_features(out, out1, gray, gray1)

affine_xform = compute_fully_projective_xform.compute_proj_xform(index, out, out1, gray, gray1)

dst = cv2.warpPerspective(pic1, affine_xform, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((pic, dst), axis=1)
cv2.imwrite("alignment_leuven13_p.jpg",final)

newImage = cv2.warpPerspective(pic, affine_xform, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic1
cv2.imwrite("stitching_leuven13_p.jpg",newImage)



pic = cv2.imread('graf1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('graf2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results, index = match_features.match_features(out, out1, gray, gray1)

affine_xform = compute_fully_projective_xform.compute_proj_xform(index, out, out1, gray, gray1)

dst = cv2.warpPerspective(pic1, affine_xform, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((pic, dst), axis=1)
cv2.imwrite("alignment_graf12_p.jpg",final)

newImage = cv2.warpPerspective(pic, affine_xform, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic1
cv2.imwrite("stitching_graf12_p.jpg",newImage)





pic = cv2.imread('graf1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('graf3.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results, index = match_features.match_features(out, out1, gray, gray1)

affine_xform = compute_fully_projective_xform.compute_proj_xform(index, out, out1, gray, gray1)

dst = cv2.warpPerspective(pic1, affine_xform, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((pic, dst), axis=1)
cv2.imwrite("alignment_graf13_p.jpg",final)

newImage = cv2.warpPerspective(pic, affine_xform, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic1
cv2.imwrite("stitching_graf13_p.jpg",newImage)



pic = cv2.imread('wall1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1_temp = cv2.imread('wall2.png')
pic1 = cv2.resize(pic1_temp, (1000, 700))
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results, index = match_features.match_features(out, out1, gray, gray1)

affine_xform = compute_fully_projective_xform.compute_proj_xform(index, out, out1, gray, gray1)

dst = cv2.warpPerspective(pic1, affine_xform, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((pic, dst), axis=1)
cv2.imwrite("alignment_wall12_p.jpg",final)

newImage = cv2.warpPerspective(pic, affine_xform, (pic1.shape[1]+100,pic1.shape[0]+100))
newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic1
cv2.imwrite("stitching_wall12_p.jpg",newImage)




pic = cv2.imread('wall1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1_temp = cv2.imread('wall3.png')
pic1 = cv2.resize(pic1_temp, (1000, 700))
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

results, index = match_features.match_features(out, out1, gray, gray1)

affine_xform = compute_fully_projective_xform.compute_proj_xform(index, out, out1, gray, gray1)

dst = cv2.warpPerspective(pic1, affine_xform, (pic1.shape[1], pic1.shape[0]))

final = np.concatenate((pic, dst), axis=1)
cv2.imwrite("alignment_wall13_p.jpg",final)

newImage = cv2.warpPerspective(pic, affine_xform, (pic1.shape[1]+100,pic1.shape[0]+100))
# newImage[0:pic1.shape[0],0:pic1.shape[1]] = 0.5*newImage[0:pic1.shape[0],0:pic1.shape[1]]+0.5 * pic1
cv2.imwrite("stitching_wall13_p.jpg",newImage)




#############        sift         ##############


pic = cv2.imread('bikes1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('bikes2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

a = ssift_descriptor.ssift_descriptor(out,gray)
b = ssift_descriptor.ssift_descriptor(out1,gray1)
dic = ssift_descriptor.ratio_test(a,b)

image = np.concatenate((pic,pic1),axis= 1)
offSet = pic1.shape[1]
for i in dic:
    if(dic[i]==0):continue

    cv2.line(image,(i[1],i[0]),(dic[i][1]+offSet,dic[i][0]),(0,255,0))
    cv2.circle(image,(i[1],i[0]),5,(255,0,0))
    cv2.circle(image,(dic[i][1]+offSet,dic[i][0]),5,(255,0,0))


cv2.imwrite("sift_bikes12.jpg",image )



pic = cv2.imread('bikes1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('bikes3.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

a = ssift_descriptor.ssift_descriptor(out,gray)
b = ssift_descriptor.ssift_descriptor(out1,gray1)
dic = ssift_descriptor.ratio_test(a,b)

image = np.concatenate((pic,pic1),axis= 1)
offSet = pic1.shape[1]
for i in dic:
    if(dic[i]==0):continue

    cv2.line(image,(i[1],i[0]),(dic[i][1]+offSet,dic[i][0]),(0,255,0))
    cv2.circle(image,(i[1],i[0]),5,(255,0,0))
    cv2.circle(image,(dic[i][1]+offSet,dic[i][0]),5,(255,0,0))


cv2.imwrite("sift_bikes13.jpg",image )




pic = cv2.imread('leuven1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('leuven2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

a = ssift_descriptor.ssift_descriptor(out,gray)
b = ssift_descriptor.ssift_descriptor(out1,gray1)
dic = ssift_descriptor.ratio_test(a,b)

image = np.concatenate((pic,pic1),axis= 1)
offSet = pic1.shape[1]
for i in dic:
    if(dic[i]==0):continue

    cv2.line(image,(i[1],i[0]),(dic[i][1]+offSet,dic[i][0]),(255,0,0))
    cv2.circle(image,(i[1],i[0]),5,(0,0,255))
    cv2.circle(image,(dic[i][1]+offSet,dic[i][0]),5,(0,0,255))

cv2.imwrite("sift_leuven12.jpg",image )




pic = cv2.imread('leuven1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('leuven3.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

a = ssift_descriptor.ssift_descriptor(out,gray)
b = ssift_descriptor.ssift_descriptor(out1,gray1)
dic = ssift_descriptor.ratio_test(a,b)

image = np.concatenate((pic,pic1),axis= 1)
offSet = pic1.shape[1]
for i in dic:
    if(dic[i]==0):continue

    cv2.line(image,(i[1],i[0]),(dic[i][1]+offSet,dic[i][0]),(0,255,0))
    cv2.circle(image,(i[1],i[0]),5,(255,0,0))
    cv2.circle(image,(dic[i][1]+offSet,dic[i][0]),5,(255,0,0))



cv2.imwrite("sift_leuven13.jpg",image )






pic = cv2.imread('graf1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('graf2.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

a = ssift_descriptor.ssift_descriptor(out,gray)
b = ssift_descriptor.ssift_descriptor(out1,gray1)
dic = ssift_descriptor.ratio_test(a,b)

image = np.concatenate((pic,pic1),axis= 1)
offSet = pic1.shape[1]
for i in dic:
    if(dic[i]==0):continue

    cv2.line(image,(i[1],i[0]),(dic[i][1]+offSet,dic[i][0]),(0,255,0))
    cv2.circle(image,(i[1],i[0]),5,(255,0,0))
    cv2.circle(image,(dic[i][1]+offSet,dic[i][0]),5,(255,0,0))


cv2.imwrite("sift_graf12.jpg",image )





pic = cv2.imread('graf1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1 = cv2.imread('graf3.png')
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

a = ssift_descriptor.ssift_descriptor(out,gray)
b = ssift_descriptor.ssift_descriptor(out1,gray1)
dic = ssift_descriptor.ratio_test(a,b)

image = np.concatenate((pic,pic1),axis= 1)
offSet = pic1.shape[1]
for i in dic:
    if(dic[i]==0):continue

    cv2.line(image,(i[1],i[0]),(dic[i][1]+offSet,dic[i][0]),(0,255,0))
    cv2.circle(image,(i[1],i[0]),5,(255,0,0))
    cv2.circle(image,(dic[i][1]+offSet,dic[i][0]),5,(255,0,0))


cv2.imwrite("sift_graf13.jpg",image )



pic = cv2.imread('wall1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1_temp = cv2.imread('wall2.png')
pic1 = cv2.resize(pic1_temp, (1000, 700))
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

a = ssift_descriptor.ssift_descriptor(out,gray)
b = ssift_descriptor.ssift_descriptor(out1,gray1)
dic = ssift_descriptor.ratio_test(a,b)

image = np.concatenate((pic,pic1),axis= 1)
offSet = pic1.shape[1]
for i in dic:
    if(dic[i]==0):continue

    cv2.line(image,(i[1],i[0]),(dic[i][1]+offSet,dic[i][0]),(0,255,0))
    cv2.circle(image,(i[1],i[0]),5,(255,0,0))
    cv2.circle(image,(dic[i][1]+offSet,dic[i][0]),5,(255,0,0))



cv2.imwrite("sift_wall12.jpg",image )



pic = cv2.imread('wall1.png')
out = detect_features.detect_features(pic)
gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

pic1_temp = cv2.imread('wall3.png')
pic1 = cv2.resize(pic1_temp, (1000, 700))
out1 = detect_features.detect_features(pic1)
gray1 = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)

a = ssift_descriptor.ssift_descriptor(out,gray)
b = ssift_descriptor.ssift_descriptor(out1,gray1)
dic = ssift_descriptor.ratio_test(a,b)

image = np.concatenate((pic,pic1),axis= 1)
offSet = pic1.shape[1]
for i in dic:
    if(dic[i]==0):continue

    cv2.line(image,(i[1],i[0]),(dic[i][1]+offSet,dic[i][0]),(0,255,0))
    cv2.circle(image,(i[1],i[0]),5,(255,0,0))
    cv2.circle(image,(dic[i][1]+offSet,dic[i][0]),5,(255,0,0))



cv2.imwrite("sift_wall3.jpg",image )