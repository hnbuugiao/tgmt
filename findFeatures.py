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
from sklearn.preprocessing import StandardScaler


# Lấy path của tập train
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# lấy tên class của tập train và đưa vào list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

#Save tất cả path của image vào listdir
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Tạo SIFT
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# tạo list chứa keypoint và descriptor
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))


# Đổi thành vertical numpy
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))


# Biểu diễn kmean
k = 100
voc, variance = kmeans(descriptors, k, 1)
count = 1

# Tạo histogram
im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

#Scaling
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# huấn luyện LinearSVC
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)
