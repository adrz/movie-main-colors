#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

def process_movie(file_path=''):
    '''
    Process movie file
    '''
    cap = cv2.VideoCapture(file_path)

    cnt = 0
    while cap.isOpened():
        success, img = cap.read()
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print(img_hsv)
        cnt+=1
        if cnt>5000:
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

    # args.input_file
    df = pickle.load(open(args.input_file, 'rb'))

if __name__ == "__main__":
    main(sys.argv[1:])

