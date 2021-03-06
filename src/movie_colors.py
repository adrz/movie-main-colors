#!/USSR/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from functools import partial
from time import time
import math


# dictionnary to convert between rgb and each colorspaces
hash_colorspace = {'hsv': [cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2RGB],
                   'luv': [cv2.COLOR_BGR2LUV, cv2.COLOR_LUV2RGB],
                   'hls': [cv2.COLOR_BGR2HLS, cv2.COLOR_HLS2RGB],
                   'xyz': [cv2.COLOR_BGR2XYZ, cv2.COLOR_XYZ2RGB],
                   'lab': [cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2RGB],
                   }


def get_kmeans_prc(img, n_clusters=3, n_jobs=8):
    """ compute the kmeans of colors and sort it according
    to its percentage
    """
    model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, max_iter=20)
    model.fit_predict(img)
    un, cnt = np.unique(model.labels_, return_counts=True)
    order_col = np.argsort(cnt)
    order_col = order_col[::-1]
    cnt = cnt[order_col]/np.sum(cnt)*100
    cols = model.cluster_centers_[order_col, :]
    return (cols, cnt)


def get_kmeans(img, n_clusters=3, n_jobs=8):
    """ compute kmeans (sklearn) """
    model = KMeans(n_clusters=n_clusters, n_jobs=n_jobs, max_iter=20)
    model.fit_predict(img)
    return model.cluster_centers_


def get_kmeans_cv_prc(img, n_clusters=3):
    """ compute opencv kmean implementation """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(np.float32(img),
                                              n_clusters, None,
                                              criteria, 10, flags)
    un, cnt = np.unique(labels, return_counts=True)
    order_col = np.argsort(cnt)
    order_col = order_col[::-1]
    cnt = cnt[order_col]/np.sum(cnt)*100
    cols = centers[order_col, :]
    return (cols, cnt)


def get_gaussian(img, n_clusters=3):
    """ compute clustering by means of gaussian mixture """
    model = GaussianMixture(n_components=n_clusters)
    model.fit(img)
    return model.means_


def hsv_to_rgb(center, colorspace='luv'):
    """ color convertion """
    if colorspace == 'hsv':
        return cv2.cvtColor(np.uint8([center]), cv2.COLOR_HSV2RGB)
    elif colorspace == 'luv':
        return cv2.cvtColor(np.uint8([center]), cv2.COLOR_LUV2RGB)


def color_to_rgb(center, colorspace='luv'):
    """ colors conversion """
    return cv2.cvtColor(np.uint8([center]), hash_colorspace[colorspace][1])


def increase_saturation(colors, ratio=1.4):
    """ increase saturation to get more 'vivid' results """
    colors_hsv = cv2.cvtColor(np.uint8(colors), cv2.COLOR_RGB2HSV)[0]
    s = colors_hsv[:, 1]
    s = np.clip(s*ratio, 0, 255)
    s = np.uint8(s)
    colors_hsv[:, 1] = s
    return cv2.cvtColor(np.uint8([colors_hsv]), cv2.COLOR_HSV2RGB)


def make_pie(sizes, cols, radius=1):
    """ plot a pie chart """
    col = [[i/255 for i in c] for c in cols]
    plt.axis('equal')
    outside, _ = plt.pie(sizes, counterclock=False,
                         radius=radius, colors=col, startangle=90)


def get_donut_chart(centers_hsv, colorspace=''):
    """ plot 3 concatenated pie chart """
    centers_rgb = np.array([hsv_to_rgb(x, colorspace) for x in centers_hsv])
    c1 = list(centers_rgb[:, 0, 0, :])
    c2 = list(centers_rgb[:, 0, 1, :])
    c3 = list(centers_rgb[:, 0, 2, :])
    len_colors = len(c1)
    e1 = (255, 255, 255)
    make_pie([1]*len_colors, c3, radius=1.2)
    make_pie([1]*len_colors, c2, radius=1)
    make_pie([1]*len_colors, c1, radius=.8)
    make_pie([1], [e1], radius=.6)
    plt.show()


def process_cols(cols_rgb, prc, blur, saturate=1):
    if saturate > 1:
        cols_rgb = [increase_saturation(x, ratio=saturate) for x in cols_rgb]
    prc_norm = [np.round(x/5) for x in prc]
    n_colors = len(prc[0])
    diff_norm = [i for i, p in enumerate(prc_norm) if np.sum(p) == 19]
    for ind in diff_norm:
        prc_norm[ind][-1] = prc_norm[ind][-1]+1
    diff_norm = [i for i, p in enumerate(prc_norm) if np.sum(p) == 21]
    for ind in diff_norm:
        prc_norm[ind][0] = prc_norm[ind][0]-1

    list_colors = []

    for p, c in zip(prc_norm, cols_rgb):
        cn = c[0]
        cc = [cn[0, :]]*int(p[0])
        n_colors = len(p)
        for i in range(1, n_colors):
            cc += [cn[i, :]]*int(p[i])
        list_colors.append(cc)
    list_colors = np.array(list_colors)
    if blur:
        list_colors = cv2.blur(list_colors, blur, cv2.BORDER_REPLICATE)
    return list_colors


def round_series_retain_integer_sum(xs, digit=0):
    new_numbers = list()
    rest = 0.0
    for n in xs:
        new_n = round(n + rest, digit)
        rest += n - new_n
        new_numbers.append(new_n)
    return new_numbers


def process_cols_2(cols_rgb, prc, blur, saturate=1):
    def lum(r, g, b):
        return np.sqrt(.241*r/255 + .691*g/255 + .068*b/255)
    if saturate > 1:
        cols_rgb = [increase_saturation(x, ratio=saturate) for x in cols_rgb]

    cols_sort = []
    prc_sort = []
    for c, p in zip(cols_rgb, prc):
        luminance = [lum(*x) for x in c[0]]
        idx_sort = sorted(range(len(luminance)), key=lambda k: luminance[k])
        c_sort = np.array([list(c[0][k]) for k in idx_sort])
        p_sort = np.array([p[k] for k in idx_sort])
        prc_sort.append(p_sort)
        cols_sort.append([c_sort])

    prc = prc_sort
    cols_rgb = cols_sort
    import pickle
    pickle.dump({'prc': prc, 'cols_rgb': cols_rgb},
                open('test.pkl', 'wb'))
    # let's be sure that sum(x) = 100 for all x in prc
    prc = [x/sum(x)*100 for x in prc]
    # hackish because
    # sum(x) is not necessary equal to sum(round(x))
    # prc_norm = [np.round(x*100) for x in prc]
    prc_norm = [round_series_retain_integer_sum(x*100)
                for x in prc]
    n_colors = len(prc[0])
    list_colors = []

    for p, c in zip(prc_norm, cols_rgb):
        cn = c[0]
        cc = [cn[0, :]]*int(p[0])
        n_colors = len(p)
        for i in range(1, n_colors):
            cc += [cn[i, :]]*int(p[i])
        list_colors.append(cc)
    list_colors = np.array(list_colors)
    if blur:
        list_colors = cv2.blur(list_colors, blur, cv2.BORDER_REPLICATE)

    return list_colors


def polarchart3(cols, prc, blur, output_file, saturate, resolution):
    cols_rgb = process_cols_2(cols, prc, blur, saturate)
    img_polar = convert2polar(cols_rgb, resolution)
    print('output_file: {}'.format(output_file))
    cv2.imwrite(output_file, img_polar)


def barchart(cols, prc, blur, output_file, saturate, resolution):
    cols_rgb = process_cols_2(cols, prc, blur, saturate)
    print('output_file: {}'.format(output_file))
    print('shape[0]: {}, shape[1]: {}'.format(
        cols_rgb.shape[0], cols_rgb.shape[1]))
    dim = (int(resolution/2), int(resolution))
    img = cv2.resize(cols_rgb, dim, interpolation=cv2.INTER_AREA)
    img = np.rot90(img, k=1)

    cv2.imwrite(output_file, img[:, :, ::-1])


def polarchart2(cols, prc, blur, output_file, saturate):
    def plt_bar(i, cols_rgb, left_outer):
        col_bb = cols_rgb[:, i, :]
        time_length = cols_rgb.shape[0]
        bot = 25-i
        # hackish way of avoiding aliasing with matplotlib
        # https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get
        # -only-fills?noredirect=1&lq=1
        ax.bar(left=left_outer,
               width=2 * np.pi / time_length, bottom=bot, color=col_bb/255.,
               linewidth=0, alpha=1, antialiased=True, rasterized=True,
               height=np.zeros_like(left_outer) + 1)
        ax.bar(left=left_outer,
               width=2 * np.pi / time_length, bottom=bot, color=col_bb/255.,
               linewidth=0, alpha=1, antialiased=True, rasterized=True,
               height=np.zeros_like(left_outer) + 1)
        ax.bar(left=left_outer,
               width=2 * np.pi / time_length, bottom=bot, color=col_bb/255.,
               linewidth=0, alpha=1, antialiased=True, rasterized=True,
               height=np.zeros_like(left_outer) + 1)
        return ax

    cols_rgb = process_cols(cols, prc, blur, saturate)
    max_pixel = 3840
    max_pixel = 1000
    r = min(1, max_pixel/cols_rgb.shape[0])
    dim = (int(cols_rgb.shape[1]), int(cols_rgb.shape[0]*r))
    cols_rgb = cv2.resize(cols_rgb, dim, interpolation=cv2.INTER_AREA)
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(polar=True))
    left_outer = np.arange(0.0, 2 * np.pi, 2 * np.pi / cols_rgb.shape[0])
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)

    partial_plt_bar = partial(
        plt_bar, cols_rgb=cols_rgb, left_outer=left_outer)

    t = time()
    _ = list(map(partial_plt_bar, range(20)))
    print(time()-t)
    ax.set_axis_off()
    plt.savefig(output_file,
                bbox_inches='tight', transparent=True)


def polar2cartesian(x, y, H, h, L, b):
    x -= .5*H
    y -= .5*H
    atan = -math.pi + np.arctan2(x, y)
    if atan < 0:
        atan += 2*math.pi
    x2 = atan*L/2/math.pi
    # if x2 < 0:
    #     x2 += math.pi*2
    y2 = (math.sqrt(x**2 + y**2))*(h)*(2/H)
    return x2, y2


def convert2polar(img, H):
    arr = np.zeros((img.shape[0], int(img.shape[1]*.15), 3))
    img = np.hstack((arr, img))

    img_polar = np.zeros((H, H, 3))
    h, L = img.shape[1], img.shape[0]
    for x in range(H):
        for y in range(H):
            x2, y2 = (int(xy)
                      for xy in np.round(polar2cartesian(
                              x, y, H, h, L, 2)))
            if (0 <= x2 < L) & (0 <= y2 < h):
                img_polar[x, y, :] = img[x2, y2, :]
    img_polar_cv = np.rot90(img_polar, k=-1)[:, :, ::-1]
    return img_polar_cv
# cv2.imwrite('out.png', cv2.cvtColor(img_polar, cv2.COLOR_RGB2BGR))


def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def process_movie(file_path='', alg='cv',
                  n_clusters=3, output_file='',
                  colorspace='luv',
                  normalize=1,
                  r=0.1):
    """ Process movie file every 10 images:

    Args:
    - file_path (str): path to movie file
    - alg (str): algorithm used for the extraction
    of the main colors. Choice between
        + cv: opencv implementation of K-Means
        + sklearn: sklearn implementation of K-Means
        + gaussian: sklearn implementation of Mixture Gaussian
    - n_clusters: number of color to be extracted
    - colorspace: colorspace used to compute the distance
                  choice between luv, hsl, lab
    """
    cap = cv2.VideoCapture(file_path)
    n_imgs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt_total = 0
    cnt = 1
    list_centers = []
    list_prc = []
    scaler = StandardScaler()
    # capture 1 out of 10 frame and processed it
    while cap.isOpened():
        while cnt % 10:
            success, img = cap.read()
            if not success:
                break
            else:
                dim = (int(img.shape[1]*r), int(img.shape[0] * r))
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                img = autocrop(img)
                if len(img) == 0:
                    success = False
                    break
            cnt += 1
            cnt_total += 1
        cnt = 1
        if success:
            if colorspace in hash_colorspace.keys():
                img = cv2.cvtColor(img, hash_colorspace[colorspace][0])
            else:
                print('wrong colorspace')
                break
        else:
            break

        # flatten the image tensor into a 2D matrix (n_pixels x 3)
        img = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

        # kmeans assumes that the data are a distributed as a mixture
        # of gaussian with covariance proportional to diagonal matrices
        # scaling it often help with this assumption
        if normalize:
            try:
                scaler.fit(img)
                img = scaler.transform(img)
            except:
                continue

        # find out the opencv kmeans implementation is the fastest
        try:
            if alg == 'cv':
                centers, prc = get_kmeans_cv_prc(img, n_clusters)
            elif alg == 'sklearn':
                centers, prc = get_kmeans_prc(img, n_clusters)
            elif alg == 'gaussian':
                centers, prc = get_gaussian(img, n_clusters)
                print(cnt_total/n_imgs*100)
            if normalize:
                centers = scaler.inverse_transform(centers)
            list_centers.append(centers)
            list_prc.append(prc)
        except:
            continue

    list_centers = [color_to_rgb(x, colorspace) for x in list_centers]
    cap.release()
    return list_centers, list_prc
