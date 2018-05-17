#!/usr/local/bin/python2.7
# -*- coding: utf-8 -*-
import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
import random

# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# Lấy path của tập train
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

#Save tất cả path của image vào listdir
image_paths = []
image_classes = []
class_temp = []
class_id = 0
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print "No such directory {}\nCheck if the file exists".format(test_path)
        exit()
    for testing_name in testing_names:
        class_temp.append(testing_name)
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_classes+=[class_id]*len(class_path)
        image_paths+=class_path
        class_id+=1
else:
    image_paths = [args["image"]]

class_id_temp = 0
for i in range(len(image_classes)):
    if i != len(image_classes)-1:
        if image_classes[i] == image_classes[i+1]:
            image_classes[i] = class_temp[class_id_temp]
        else:
            image_classes[i] = class_temp[class_id_temp]
            class_id_temp+=1
    else:
        image_classes[i] = class_temp[class_id_temp]

# Tạo SIFT
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# tạo list chứa keypoint và descriptor
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    #if im.any():
        #print "No such file {}\nCheck if the file exists".format(image_path)
        #exit()
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))

# Đổi thành vertical numpy
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))

# tạo histogram
test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale
test_features = stdSlr.transform(test_features)

# Dự đoán
predictions =  [classes_names[i] for i in clf.predict(test_features)]
if len(predictions) > 1:
    print(predictions)
    count1 = len(predictions)
    count2 = 0
    for i in range(len(predictions)):
        if predictions[i] == image_classes[i]:
            count2+=1

    count3 = float(count2)/count1
    print(count)
else:
    pushbacktodata = "".join(predictions)
    temp = "dataset/trainnew/"
    temp2 = temp+pushbacktodata

    dir = imutils.imlist(temp2)
    dir3 = []
    dir3.append(random.choice(dir))
    while len(dir3) < 3:
        dir2 = random.choice(dir)
        count = 0
        for i in range(len(dir3)):
            if dir2 == dir3[i]:
                count+=1
        if count == 0:
            dir3.append(dir2)


    for path in dir3:
        image = cv2.imread(path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", image)
        cv2.waitKey(3000)




'''
# Visualize the results, if "visualize" flag set to true by the user
if args["visualize"]:
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        cv2.imshow("Image", image)
        cv2.waitKey(3000)
'''
