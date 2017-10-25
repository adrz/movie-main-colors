#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import argparse
from sklearn.cluster import KMeans
import pickle
from libKMCUDA import kmeans_cuda
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture



def process_cols(cols_rgb, prc, blur, saturate=1):
    if saturate>1:
        cols_rgb = [increase_saturation(x, ratio=saturate) for x in cols_rgb]
    prc_norm = [np.round(x/5) for x in prc]
    n_colors = len(prc[0])
    diff_norm = [i for i,p in enumerate(prc_norm) if np.sum(p) == 19]
    for ind in diff_norm:
        prc_norm[ind][-1] = prc_norm[ind][-1]+1
    diff_norm = [i for i,p in enumerate(prc_norm) if np.sum(p) == 21]
    for ind in diff_norm:
        prc_norm[ind][0] = prc_norm[ind][0]-1

    list_colors = []

    for p,c in zip(prc_norm,cols_rgb):
        cn = c[0]
        cc = [cn[0,:]]*int(p[0])
        n_colors = len(p)
        for i in range(1, n_colors):
            cc+=[cn[i,:]]*int(p[i])
        list_colors.append(cc)
    list_colors = np.array(list_colors)
    if blur:
        list_colors = cv2.blur(list_colors, blur, cv2.BORDER_REPLICATE)
    return list_colors

        
def increase_saturation(colors, ratio=1.4):
    colors_hsv = cv2.cvtColor(np.uint8(colors), cv2.COLOR_RGB2HSV)[0]
    s = colors_hsv[:,1]
    s = np.clip(s*ratio, 0, 255)
    s = np.uint8(s)
    colors_hsv[:,1] = s
    return cv2.cvtColor(np.uint8([colors_hsv]), cv2.COLOR_HSV2RGB)

def make_pie(sizes, cols, radius=1):
    col = [[i/255 for i in c] for c in cols]
    plt.axis('equal')
    outside, _ = plt.pie(sizes, counterclock=False, radius=radius, colors=col, startangle=90)

def polarchart(cols_rgb, prc, blur=True, output_file='', saturate=1):
    '''
    polarchart for main colors in movie
    '''
    list_colors = process_cols(cols_rgb, prc, blur, saturate)
    len_time = len(list_colors)

    for i in range(20):
        radius = 1-.04*i
        c = list(list_colors[:,i,:])
        make_pie([1]*len_time, c, radius=radius)

    make_pie([1], [(255,255,255)], radius=1-.04*20)
    plt.savefig(output_file, dpi=300)    



def polarchart2(cols, prc, blur, output_file, saturate):
    cols_rgb = process_cols(cols, prc, blur, saturate)
    max_pixel = 3840
    r   = max_pixel/cols_rgb.shape[0]
    dim = (int(cols_rgb.shape[1]), int(cols_rgb.shape[0]*r))
    cols_rgb = cv2.resize(cols_rgb, dim, interpolation=cv2.INTER_AREA)
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(polar=True))
    time, n_colors = cols_rgb.shape[0], cols_rgb.shape[1]
    left_outer = np.arange(0.0, 2 * np.pi, 2 * np.pi / time)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)

    def plt_bar(i, cols_rgb, left_outer):
        col_bb = cols_rgb[:,i,:]
        bot = 25-i
        ax.bar(left=left_outer,
               width=2 * np.pi / time, bottom=bot, color=col_bb/255.,
               linewidth=0, alpha=1, antialiased=True,
               height=np.zeros_like(left_outer) + 1)
        return ax

    from functools import partial
    from multiprocessing import Pool
    partial_plt_bar = partial(plt_bar,cols_rgb=cols_rgb, left_outer=left_outer)
    from time import time as tt
    
    t = tt()
    x = list(map(partial_plt_bar, range(20)))
    print(tt()-t)

    t = tt()
    x = list(map(partial_plt_bar, range(20)))
    print(tt()-t)

    t = tt()
    x = list(map(partial_plt_bar, range(20)))
    print(tt()-t)

    #, edgecolor=None

    # Add a problem with antialiasing
    # https://stackoverflow.com/questions/15822159/aliasing-when-saving-matplotlib-filled-contour-plot-to-pdf-or-eps
    # https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills/32911283#32911283
    # t = tt()
    # for i in range(20):
    #     col_bb = cols_rgb[:,i,:]
    #     bot = 25-i
    #     ax.bar(left=left_outer,
    #            width=2 * np.pi / time, bottom=bot, color=col_bb/255.,
    #            linewidth=0, alpha=1, antialiased=True,
    #            height=np.zeros_like(left_outer) + 1)
    # print(tt()-t)
    
    # for i in range(20):
    #     col_bb = cols_rgb[:,i,:]
    #     bot = 25-i
    #     ax.bar(left=left_outer,
    #            width=2 * np.pi / time, bottom=bot, color= col_bb/255.,
    #            linewidth=0, alpha=1, antialiased=True,
    #            height=np.zeros_like(left_outer) + 1)

    # for i in range(20):
    #     col_bb = cols_rgb[:,i,:]
    #     bot = 25-i
    #     ax.bar(left=left_outer,
    #            width=2 * np.pi / time, bottom=bot, color= col_bb/255.,
    #            linewidth=0, alpha=1, antialiased=True,
    #            height=np.zeros_like(left_outer) + 1)

    ax.set_axis_off()
    plt.savefig(output_file, \
                bbox_inches='tight', transparent=True)



def barchart(cols, prc, blur, output_file, saturate):
    '''
    Barchart for main colors in movie
    '''
    
    img = process_cols(cols, prc, blur, saturate)
    dim = (int(img.shape[0]*.2), int(img.shape[0]))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = cv2.transpose(img)
    plt.imshow(img)
    a = plt.gcf().gca()
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])
    plt.axis('off')
    plt.savefig('test_bar.png',dpi=500, \
               bbox_inches='tight', edgecolor=None, \
                pad_inches=0, transparent=True)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help="input movie file",
                        default="movie.mp4")
    parser.add_argument('-x', '--blur_x',
                        help="", type=int,
                        default=5)
    parser.add_argument('-y', '--blur_y',
                        help="", type=int,
                        default=5)
    parser.add_argument('-s', '--saturate', type=float,
                        help='',
                        default=1)
    parser.add_argument('-t', '--type',
                        help='polar or bar',
                        default='polar')
    parser.add_argument('-o', '--output_file',
                        help="image output",
                        default='')
    args = parser.parse_args()
    dt = pickle.load(open(args.input_file,'rb'))
    cols = dt['centers']
    prc = dt['prc']
    if args.blur_x!=0:
        blur = (args.blur_x, args.blur_y)
    else:
        blur = False

    if args.type == 'polar':
        polarchart2(cols=cols, prc=prc, \
                   blur=blur, output_file=args.output_file, \
                   saturate=args.saturate)
    elif args.type == 'bar':
        barchart(cols_rgb=cols, prc=prc,\
                 blur=blur, output_file=args.output_file, \
                 saturate=args.saturate)

if __name__ == "__main__":
    main(sys.argv[1:])
