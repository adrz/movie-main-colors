#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import argparse
from sklearn.cluster import KMeans
import scipy
import pickle

def get_kmeans(img=img, n_clusters=3, n_jobs=8):
    model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, max_iter=10)
    model.fit_predict(img)
    return model.cluster_centers_

def get_kmeans_cv(img=img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(np.float32(img),3,None,criteria,10,flags)
    return centers

import time
a1 = time.time()
r1 = get_kmeans()
a2 = time.time()

b1 = time.time()
r2 = get_kmeans_cv()
b2 = time.time()

print('t1: %f ; t2: %f'%(a2-a1,b2-b1))

def process_movie(file_path=''):
    '''
    Process movie file
    '''
    cap = cv2.VideoCapture(file_path)
    n_imgs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt_total = 0
    cnt = 1
    list_centers = []
    while cap.isOpened():
        while cnt%10:
            success, img = cap.read()
            cnt+=1
            cnt_total+=1
        cnt=1
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv = scipy.misc.imresize(img_hsv, .1)
        img_hsv = img_hsv.reshape(img_hsv.shape[0]*img_hsv.shape[1], img_hsv.shape[2])
        list_centers.append(get_kmeans(img_hsv, 3, 6))
        print(cnt_total/n_imgs*100)
        if cnt>1000:
            break
    cap.release()
    pickle.dump(list_centers, open('data.p','wb'))

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help="input movie file",
                        default="movie.mp4")
    parser.add_argument('-o', '--output_file',
                        help="image output",
                        default='output.pdf')
    args = parser.parse_args()
    process_movie(file_path=args.input_file)
    # args.input_file


if __name__ == "__main__":
    main(sys.argv[1:])

