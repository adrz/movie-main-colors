#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import argparse
from sklearn.cluster import KMeans

def get_kmeans(img, n_clusters, n_jobs=6):
    model = KMeans(n_clusters=n_clusters, n_jobs=6)
    model.fit_predict(img)
    return model.cluster_centers_

def process_movie(file_path=''):
    '''
    Process movie file
    '''
    cap = cv2.VideoCapture(file_path)

    cnt = 0
    list_centers = []
    while cap.isOpened():
        success, img = cap.read()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
        list_centers+=get_kmeans(img_hsv, 3, 6)
        cnt+=1
        if cnt>1000:
            break
    cap.release()

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

