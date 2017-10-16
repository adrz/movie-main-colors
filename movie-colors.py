#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import argparse
from sklearn.cluster import KMeans
import scipy
import pickle
from libKMCUDA import kmeans_cuda

def get_kmeans(img, n_clusters=3, n_jobs=8):
    model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, max_iter=20)
    model.fit_predict(img)
    return model.cluster_centers_

def get_kmeans_cv(img, n_clusters=3):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(np.float32(img), \
                                            n_clusters, None,\
                                            criteria,10,flags)
    return centers


def get_kmeans_cuda(img, n_clusters=3):
#    centroids, assignments, avg_distance = kmeans_cuda(
    #    arr, 4, metric="cos", verbosity=1, seed=3, average_distance=True)
    centroids, assignments = kmeans_cuda(np.float32(img), n_clusters, verbosity=1, yinyang_t=0)
    return centroids



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
            if success==False:
                break
            cnt+=1
            cnt_total+=1
        cnt=1
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv = scipy.misc.imresize(img_hsv, .1)
        img_hsv = img_hsv.reshape(img_hsv.shape[0]*img_hsv.shape[1], img_hsv.shape[2])
#        list_centers.append(get_kmeans_cv(img_hsv, 3))
        list_centers.append(get_kmeans_cuda(img_hsv, 3))
        print(cnt_total/n_imgs*100)
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

