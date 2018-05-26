#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import subprocess
import os


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help="full path of input movie file",
                        default="/home/user/movie.mp4")
    parser.add_argument('-a', '--alg',
                        help="kmeans implementation choice: sklearn, cv, cuda, gaussian",
                        default="cv")
    parser.add_argument('-c', '--colorspace',
                        help="colorspace to compute clusters (hsv/hls/luv/lab)",
                        default="lab")
    parser.add_argument('-n', '--n_colors',
                        help='number of colors to extract',
                        default="3")
    parser.add_argument('--normalize',
                        default="1")
    parser.add_argument('-x', '--blur_xy',
                        help="",
                        default="5")
    parser.add_argument('-s', '--saturate',
                        help='',
                        default="1")
    parser.add_argument('-r', '--resolution', type=int,
                        help='resolution of the output in pixel',
                        default=3000)
    parser.add_argument('-t', '--type',
                        help='polar or bar',
                        default='polar')
    parser.add_argument('-o', '--output_file',
                        help="image output (will be saved in results/)",
                        default='output.png')
    args = parser.parse_args()
    path_file = os.path.dirname(args.input_file)
    base_file = os.path.basename(args.input_file)
    # subprocess.check_call(['docker', 'build', '-t', 'moviecolors', '.'])
    subprocess.check_call(['docker', 'run', '--rm',
                           '-v', '{}:/data'.format(path_file),
                           '-v', '{}/results:/results'.format(os.getcwd()),
                           'adrz/movie-colors:latest',
                           '/data/{}'.format(base_file),
                           args.alg, args.colorspace,
                           args.n_colors, args.normalize,
                           args.blur_xy, args.saturate,
                           args.resolution,
                           args.type,
                           '/results/{}'.format(args.output_file)])


if __name__ == "__main__":
    main(sys.argv[1:])
