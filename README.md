# Feature-Match
Computer Vision 
In this folder, it contains some feature match function examples.
The diver.py will run all those functions using the correspond images in the image folder.

python version: 2.7
Libary: numpy, cv2, matplotlib, math, random

detect_features.py will detect the features in images using Harris Corner Detection and Non-Maximal suppression.

match_features.py uses SSD to matches the features returned by detect_features.py. Mutual marriage methon is used to improve the perfermance.

nonmaxsuppts.py is the Non-Maximal suppression function.

compute_affine_xform.py and compute_fully_projective_xform.py will calculate the affine transformation matrix and projective matrix of two pictures.

ssift_descriptor.py uses sift way to do features match.