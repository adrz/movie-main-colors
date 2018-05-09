#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from src.movie_colors import (process_movie,
                              polarchart3,
                              polarchart2,
                              barchart)
import argcomplete


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help="input movie file",
                        default="movie.mp4")
    parser.add_argument('-a', '--alg',
                        help="kmeans implementation choice: sklearn, cv, cuda, gaussian",
                        default="cv")
    parser.add_argument('-c', '--colorspace',
                        help="colorspace to compute clusters (hsv/hls/luv/lab)",
                        default="lab")
    parser.add_argument('-n', '--n_colors', type=int,
                        help='number of colors to extract',
                        default=3)
    parser.add_argument('--normalize', type=int,
                        default=1)
    parser.add_argument('-x', '--blur_xy',
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
                        default='output.pdf')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    cols, prc = process_movie(file_path=args.input_file,
                              alg=args.alg,
                              normalize=args.normalize,
                              n_clusters=args.n_colors,
                              colorspace=args.colorspace)

    import pickle
    pickle.dump({'cols': cols, 'prc': prc}, open('out.p', 'wb'))
    if args.blur_xy != 0:
        blur = (args.blur_xy, args.blur_xy)
    else:
        blur = False

    if args.type == 'polar':
        polarchart2(cols=cols, prc=prc,
                    blur=blur, output_file=args.output_file,
                    saturate=args.saturate)
    if args.type == 'polar2':
        polarchart3(cols=cols, prc=prc,
                    blur=blur, output_file=args.output_file,
                    saturate=args.saturate)
    elif args.type == 'bar':
        barchart(cols_rgb=cols, prc=prc,
                 blur=blur, output_file=args.output_file,
                 saturate=args.saturate)


if __name__ == "__main__":
    main(sys.argv[1:])
