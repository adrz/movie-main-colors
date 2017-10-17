#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import argparse
from sklearn.cluster import KMeans
import pickle
from libKMCUDA import kmeans_cuda


def make_pie(sizes, colors, radius=1):
    col = [[i/255 for i in c] for c in colors]

    plt.axis('equal')
    outside, _ = plt.pie(sizes, counterclock=False, radius=radius, colors=col, startangle=90)

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
    centroids, assignments = kmeans_cuda(np.float32(img), n_clusters, verbosity=0, yinyang_t=0)
    return centroids

def hsv_to_rgb(center):
    return cv2.cvtColor(np.uint8([center]), cv2.COLOR_HSV2RGB)

def get_donut_chart(centers_hsv):
    centers_rgb = np.array(list(map(hsv_to_rgb, centers_hsv)))
    c1 = list(centers_rgb[:,0,0,:])
    c2 = list(centers_rgb[:,0,1,:])
    c3 = list(centers_rgb[:,0,2,:])

#    c1 = list(centers_rgb[:,0,:,0])
#    c2 = list(centers_rgb[:,0,:,1])
#    c3 = list(centers_rgb[:,0,:,2])

    len_colors = len(c1)
    e1 = (255, 255, 255)
    make_pie([1]*len_colors, c1, radius=1.2)
    make_pie([1]*len_colors, c2, radius=1)
    make_pie([1]*len_colors, c3, radius=.8)
    make_pie([1], [e1], radius=.6)
    plt.show()

    


def process_movie(file_path='', alg='cv', output_file=''):
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
        if success:    
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            break
        r = .1
        dim = (int(img_hsv.shape[1]*r), int(img_hsv.shape[0] * r))
        img_hsv = cv2.resize(img_hsv, dim, interpolation=cv2.INTER_AREA)
        img_hsv = img_hsv.reshape(img_hsv.shape[0]*img_hsv.shape[1], img_hsv.shape[2])
        if alg=='cv':
            list_centers.append(get_kmeans_cv(img_hsv, 3))
        elif alg=='cuda':
            list_centers.append(get_kmeans_cuda(img_hsv, 3))
        elif alg=='sklearn':
            list_centers.append(get_kmeans(img_hsv, 3))
        print(cnt_total/n_imgs*100)
    cap.release()
    pickle.dump(list_centers, open(output_file,'wb'))

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help="input movie file",
                        default="movie.mp4")
    parser.add_argument('-a', '--alg',
                        help="kmeans implementation choice: sklearn, cv, cuda",
                        default="cv")
    parser.add_argument('-o', '--output_file',
                        help="image output",
                        default='output.pdf')
    args = parser.parse_args()
    process_movie(file_path=args.input_file, alg=args.alg, output_file=args.output_file)
    # args.input_file


if __name__ == "__main__":
    main(sys.argv[1:])



import matplotlib.pyplot as plt
import numpy as np


plt.show()