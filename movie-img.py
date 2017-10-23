#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys,os
import argparse
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from functools import reduce
from operator import mul

from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
from MulticoreTSNE import MulticoreTSNE as TSNE_multi


def cov(X):
    """
    Covariance matrix
    note: specifically for mean-centered data
    note: numpy's `cov` uses N-1 as normalization
    """
    return dot(X.T, X) / X.shape[0]

def pca(data, pc_count = None):
    """
    Principal component analysis using eigenvalues
    note: this mean-centers and auto-scales the data (in-place)
    """
    data -= mean(data, 0)
    data /= std(data, 0)
    C = cov(data)
    E, V = eigh(C)
    key = argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    # credit: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    return cv2.Laplacian(image, cv2.CV_64F).var()

def get_features(list_img_path):
    model = VGG19(weights='imagenet', include_top=False)
    features = []
    n_images = len(list_img_path)
    for idx, img_path in enumerate(list_img_path):
        print('getting features for %s %d/%d'%
              (img_path, idx+1, n_images))
        # Resize image to be 224x244
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        y = model.predict(x)
        # Vectorize the 7x7x512 tensor                                                           
        y = y.reshape(reduce(mul, y.shape, 1))
        features.append(y)
    return (list_img_path, features)


def process_movie(file_path, path_img='imgs'):
    if not os.path.exists(path_img):
        os.makedirs(directory)
    cap = cv2.VideoCapture(file_path)
    n_imgs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt_total = 0
    cnt = 1
    img_list = []
    img_list_gray = []
    list_img_path = []
    while cap.isOpened():
        while cnt%24:
            success, img = cap.read()
            if success==False:
                return list_img_path
            r = 0.5
            dim = (int(img.shape[1]*r), int(img.shape[0] * r))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_list.append(img)
            img_list_gray.append(img_gray)
#            cv2.imwrite("img/blurry/img-%d.jpg"%cnt_total, img)
            cnt+=1
            cnt_total+=1

        cnt = 1
        blurness = []
        blurness = [variance_of_laplacian(img) for img in img_list_gray]
        max_idx = np.argmax(blurness)
        img_p = "%s/img-%d.jpg"%(path_img, cnt_total)
        cv2.imwrite(img_p, img_list[max_idx])
        list_img_path.append(img_p)
        img_list = []
        img_list_gray = []
        print(cnt_total/n_imgs*100)
    return list_img_path


def tsne_to_grid(data, output):
    #https://github.com/Quasimondo/RasterFairy/blob/master/examples/Raster%20Fairy%20Demo%201.ipynb
    # https://github.com/bmcfee/RasterFairy (vpython3)

#    from sklearn.decomposition import PCA
    import rasterfairy, math

    data = np.array(data)
    
#    pca = PCA(n_components=500, whiten=True)
    n_pts = data.shape[0]
    #    data_pca = pca.fit_transform(data)
    data_pca, U, V = pca(data, pc_count=500)

    data_pca = list(data_pca)
    sqrt_npts = math.ceil(math.sqrt(n_pts))
    n_diff = sqrt_npts**2 - n_pts
    data_pca+=[data_pca[-1]]*n_diff
    data_pca = np.array(data_pca)
    list_file+=[list_file[-1]]*n_diff

    tsne = TSNE_multi(n_components=2, perplexity=50, n_jobs=8)
    data_tsne = tsne.fit_transform(data_pca.astype(np.float64))

    arrangements = rasterfairy.getRectArrangements(data_tsne.shape[0])


    img = cv2.imread(list_file[0])
    ratio = img.shape[0]/img.shape[1]
    r = 0.1
    dim = (int(img.shape[1]*r), int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    dim_x = img.shape[0]
    dim_y = img.shape[1]
    
    grid_xy, (width, height) = rasterfairy.transformPointCloud2D(data_tsne,target=arrangements[0])

    img_big = np.zeros((dim_x*width, dim_y*height, 3))

    for i, file_img in enumerate(list_file):
        xy = grid_xy[i,:]
        pos_x = int(xy[0]*dim_x)
        pos_y = int(xy[1]*dim_y)
        img = cv2.imread(file_img)
        r = 0.1
        dim = (int(img.shape[1]*r), int(img.shape[0] * r))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img_big[pos_x:pos_x+dim_x, pos_y:pos_y+dim_y,:] = img
    cv2.imwrite(output, img_big)


file_ = '/home/dude/hdd/movies/interstellar (2014) (2014) - gerald yelverton [9.0].mp4'
list_files = process_movie(file_, path_img='imgs/inter')
list_files, features = get_features(list_files)
tsne_to_grid(features, 'res/interstellar.jpg')

file_ = '/home/dude/hdd/movies/the matrix (1999) - lana wachowski [8.7].mp4'
list_files = process_movie(file_, path_img='imgs/matrix')
list_files, features = get_features(list_files)
tsne_to_grid(features, 'res/matrix.jpg')

file_ = '/home/dude/hdd/movies/spring breakers (2012) - harmony korine [5.3].mp4'
list_files = process_movie(file_, path_img='imgs/spring')
list_files, features = get_features(list_files)
tsne_to_grid(features, 'res/spring-breakers.jpg')

file_ = '/home/dude/hdd/movies/pulp fiction (1994) - quentin tarantino [8.9].mp4'
list_files = process_movie(file_, path_img='imgs/pulp')
list_files, features = get_features(list_files)
tsne_to_grid(features, 'res/pulp-fiction.jpg')

file_ = '/home/dude/hdd/movies/mad max fury road - george miller.mp4'
list_files = process_movie(file, path_img='imgs/madmax')
list_files, features = get_features(list_files)
tsne_to_grid(features, 'res/mad-mad.jpg')
