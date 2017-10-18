#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.colors as cls
import cv2
import numpy as np
import sys
import argparse
from sklearn.cluster import KMeans
import pickle
from libKMCUDA import kmeans_cuda
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

hash_colorspace = {'hsv': [cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2RGB],
                   'luv': [cv2.COLOR_BGR2LUV, cv2.COLOR_LUV2RGB],
                   'hls': [cv2.COLOR_BGR2HLS, cv2.COLOR_HLS2RGB],
                   'xyz': [cv2.COLOR_BGR2XYZ, cv2.COLOR_XYZ2RGB],
                   'lab': [cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2RGB],
}

def make_pie(sizes, cols, radius=1):
    col = [[i/255 for i in c] for c in cols]
    plt.axis('equal')
    outside, _ = plt.pie(sizes, counterclock=False, radius=radius, colors=col, startangle=90)

def get_kmeans_prc(img, n_clusters=3, n_jobs=8):
    model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, max_iter=20)
    model.fit_predict(img)
    un, cnt = np.unique(model.labels_, return_counts=True)
    order_col = np.argsort(cnt)
    order_col = order_col[::-1]
    cnt = cnt[order_col]/np.sum(cnt)*100
    cols = model.cluster_centers_[order_col,:]
    return (cols, cnt)


def get_kmeans(img, n_clusters=3, n_jobs=8):
    model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, max_iter=20)
    model.fit_predict(img)
    return model.cluster_centers_

def get_kmeans_cv_prc(img, n_clusters=3):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(np.float32(img), \
                                            n_clusters, None,\
                                            criteria,10,flags)
    un, cnt = np.unique(labels, return_counts=True)
    order_col = np.argsort(cnt)
    order_col = order_col[::-1]
    cnt = cnt[order_col]/np.sum(cnt)*100
    cols = centers[order_col,:]
    return (cols, cnt)

def get_gaussian(img, n_clusters=3):
    model = GaussianMixture(n_components=n_clusters)
    model.fit(img)
    return model.means_

def get_kmeans_cuda(img, n_clusters=3):
    centroids, assignments = kmeans_cuda(np.float32(img), n_clusters, \
                                         verbosity=0, metric="cos")
    return centroids
 
    
def hsv_to_rgb(center, colorspace='luv'):
    if colorspace=='hsv':
        return cv2.cvtColor(np.uint8([center]), cv2.COLOR_HSV2RGB)
    elif colorspace=='luv':
        return cv2.cvtColor(np.uint8([center]), cv2.COLOR_LUV2RGB)

def color_to_rgb(center, colorspace='luv'):
    return cv2.cvtColor(np.uint8([center]), hash_colorspace[colorspace][1])

        
def increase_saturation(colors, ratio=1.4):
    colors_hsv = cv2.cvtColor(np.uint8(colors), cv2.COLOR_RGB2HSV)[0]
    s = colors_hsv[:,1]
    s = np.clip(s*ratio, 0, 255)
    s = np.uint8(s)
    colors_hsv[:,1] = s
    return cv2.cvtColor(np.uint8([colors_hsv]), cv2.COLOR_HSV2RGB)


def get_donut_chart(centers_hsv, colorspace=''):
    centers_rgb = np.array([hsv_to_rgb(x, colorspace) for x in centers_hsv])
    c1 = list(centers_rgb[:,0,0,:])
    c2 = list(centers_rgb[:,0,1,:])
    c3 = list(centers_rgb[:,0,2,:])

#    c1 = list(centers_rgb[:,0,:,0])
#    c2 = list(centers_rgb[:,0,:,1])
#    c3 = list(centers_rgb[:,0,:,2])

    len_colors = len(c1)
    e1 = (255, 255, 255)
    make_pie([1]*len_colors, c3, radius=1.2)
    make_pie([1]*len_colors, c2, radius=1)
    make_pie([1]*len_colors, c1, radius=.8)
    make_pie([1], [e1], radius=.6)
    plt.show()


def polarchart(cols_rgb, prc, colorspace='luv'):
    '''
    polarchart for main colors in movie
    '''
#    cols_rgb = np.array([color_to_rgb(x, colorspace) for x in cols])
    prc_norm = [np.round(x/5) for x in prc]
    n_colors = len(prc[0])
#    diff_norm = np.where( np.sum(prc_norm, 1) == 19 )[0]
    diff_norm = [i for i,p in enumerate(prc_norm) if np.sum(p) == 19]
    for ind in diff_norm:
        prc_norm[ind][-1] = prc_norm[ind][-1]+1
    diff_norm = [i for i,p in enumerate(prc_norm) if np.sum(p) == 21]
    for ind in diff_norm:
        prc_norm[ind][0] = prc_norm[ind][0]-1

#    prc_norm = np.uint8(prc_norm)
    list_colors = []

    for p,c in zip(prc_norm,cols_rgb):
        cn = c[0]
        cc = [cn[0,:]]*int(p[0])
        n_colors = len(p)
        for i in range(1, n_colors):
            cc+=[cn[i,:]]*int(p[i])
        list_colors.append(cc)
    list_colors = np.array(list_colors)


    len_time = len(list_colors)

    for i in range(20):
        radius = 1-.04*i
        c = list(list_colors[:,i,:])
        make_pie([1]*len_time, c, radius=radius)

    make_pie([1], [(255,255,255)], radius=1-.04*20)


def barchart(cols, prc, width=2):
    '''
    Barchart for main colors in movie
    '''
    cols_norm = [[i/255 for i in c] for c in cols]
    n_colors = len(prc[0])
    for time in range(len(cols)):
        bot = 0
        print(time)
        for col in range(n_colors):
            try:
                prc_col = prc[time][col]
            except:
                prc_col = 0
            col_bar = cls.to_hex(cols_norm[time][0][col,:])
            plt.bar(time, prc_col, width, color=col_bar, bottom=bot)
            bot+=prc_col




def process_movie(file_path='', alg='cv', \
                  n_clusters=3, output_file='',
                  colorspace='luv', \
                  normalize=1, \
                  r=0.1):
    '''Process movie file every 10 images:

    Args:
    - file_path (str): path to movie file
    - alg (str): algorithm used for the extraction
    of the main colors. Choice between
        + cv: opencv implementation of K-Means
        + cuda: cuda implementation of K-Means
        + sklearn: sklearn implementation of K-Means
        + gaussian: sklearn implementation of Mixture Gaussian
    - n_clusters: number of color to be extracted
    - colorspace: colorspace used to compute the distance. Choice between luv, hsl, lab
    '''
    cap = cv2.VideoCapture(file_path)
    n_imgs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt_total = 0
    cnt = 1
    list_centers = []
    list_prc = []
    ##
    scaler = StandardScaler()
    ##
    
    while cap.isOpened():
        while cnt%10:
            success, img = cap.read()
            cnt+=1
            cnt_total+=1
        cnt=1
        if success:
            if colorspace in hash_colorspace.keys():
                img = cv2.cvtColor(img, hash_colorspace[colorspace][0])
            else:
                print('wrong colorspace')
                break
        else:
            break
        dim = (int(img.shape[1]*r), int(img.shape[0] * r))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

        if normalize:
            scaler.fit(img)
            img = scaler.transform(img)
        
        if alg=='cv':
            centers, prc = get_kmeans_cv_prc(img, n_clusters)
        elif alg=='cuda':
            centers, prc = get_kmeans_cuda(img, n_clusters)
        elif alg=='sklearn':
            centers, prc = get_kmeans_prc(img, n_clusters)
        elif alg=='gaussian':
            centers, prc = get_gaussian(img, n_clusters)
        print(cnt_total/n_imgs*100)
        if normalize:    
            centers = scaler.inverse_transform(centers)    

        list_centers.append(centers)
        list_prc.append(prc)

    list_centers = [color_to_rgb(x, colorspace) for x in list_centers]
    cap.release()
    pickle.dump({'centers': list_centers,
                 'prc': list_prc,
                 'colorspace': colorspace},
                open('data_save.p','wb'))
    polarchart(list_centers, list_prc)
    plt.savefig(output_file)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help="input movie file",
                        default="movie.mp4")
    parser.add_argument('-a', '--alg',
                        help="kmeans implementation choice: sklearn, cv, cuda, gaussian",
                        default="cv")
    parser.add_argument('-c', '--colorspace',
                        help="colorspace to compute clusters (hsv/hls/luv)",
                        default="luv")
    parser.add_argument('-n', '--n_colors', type=int,
                        help='number of colors to extract',
                        default=3)
    parser.add_argument('--normalize', type=int,
                        default=0)
    parser.add_argument('-o', '--output_file',
                        help="image output",
                        default='output.pdf')
    args = parser.parse_args()
    process_movie(file_path=args.input_file, \
                  alg=args.alg, \
                  normalize=args.normalize, \
                  n_clusters=args.n_colors,\
                  output_file=args.output_file,\
                  colorspace=args.colorspace)
    # args.input_file


if __name__ == "__main__":
    main(sys.argv[1:])


