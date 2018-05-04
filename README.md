# Yet another movie colors extractor

## Overview

This project aims to extract the main colors of each frame of a movie. It is based on a Kmeans
clustering of color. [This blog
post](http://www.alanzucconi.com/2015/05/24/how-to-find-the-main-colours-in-an-image/) do
a great job explaining the process.

Lab colorspace has been used as it is supposed to be more [perceptually uniform](https://en.wikipedia.org/wiki/Color_difference#Tolerance).

## Dependencies

* Linux (not tested on macOS, nor Windows)
* Python-3
* [OpenCV](https://opencv.org/)

## How to run it

The required computation is quick heavy. It took around 10 solid minutes to generate a
polar plot for a movie of 1h30 on an intel 7700k.

### Good old local install

I will supposed that OpenCV is properly installed on your machine. It is a painful
process ! I will not redirect you to any tutorials as those tend to be outdated
quickly. Good luck.

```bash
	# Debian/Ubuntu
	$ apt install python-3 python-pip
	$ pip install --upgrade virtualenv
	$ git clone https://github.com/adrz/movie-main-colors
	$ cd movie-main-colors
	$ virtualenv -p python3 env
	$ . env/bin/activate
	$ pip install -r requirements.txt
	$ python main-colors.py -i "/home/user/movie.mp4" -o "/home/user/output.png"
```

### Docker

First install docker: [see instructions](https://docs.docker.com/install/)

```bash
	# Debian/Ubuntu
	$ python main-colors-docker.py -i "/home/user/movie.mp3" -o output.png
```

The result will be output in the movie-main-colors/results/ folder.

## Parameters/Options

```
  -h, --help            show this help message and exit
  -i, --input_file      input movie file (movie.mp4)
  -o, --ouput_file      output image file (colors-movie.png)
  -a, --alg             algorithm used to cluster colors (choice: sklearn, cv,
                        gaussian), default: cv
  -c, --colorspace      colorspace in which the clusters are computed (choice: hsv, hls,
  luv, lab), default: lab
  -n, --n_colors        number of colors to extract, default: 3
  --normalize           normalize the colors before clustering (choice: 0, 1), default: 1
  -x, blur_xy           width of the smoothing kernel, default: 5
  -s, --saturate        color saturation factor, a value above 1 make color more ''vivid'', default: 1
  -t, --type            type of plot (choice: polar, bar), default: polar
```


## Some other movie colors extractors/info

Before jumping into other project, I highly recommend you a [very interesting analysis of
uses of color in movies](https://www.youtube.com/watch?v=tILIeNjbH1E). [Some guys on
twitter](https://twitter.com/CINEMAPALETTES) also share homemade color palettes based on a
single iconic frame of the movies.

Several movie colors extractors exist in the wild. Most of them produces a horizontal or
vertical ''barcode'' produced by a concatenation of the average color of individual
frames. It is often less computationally intensive and rely on *ffmpeg*: [movie barcodes
generator](https://github.com/timbennett/movie-barcodes/). You could even buy a poster of
your favorite movie [thecolorinmotion.com](https://thecolorsofmotion.com/films)


http://www.alanzucconi.com/2015/05/24/how-to-find-the-main-colours-in-an-image/
https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/

https://www.studiobinder.com/blog/how-to-use-color-in-film-50-examples-of-movie-color-palettes/
The most impressive work is by far a bachelor graduation project called
[cinemetrics](http://cinemetrics.fredericbrodbeck.de/).

 convert df.png -color-matrix '  1.2 -0.1 -0.1
>                                 -0.1  1.2 -0.1
>                                 -0.1 -0.1  1.2 ' dd.png
