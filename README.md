# Yet another movie colors extractor

## Overview

## Dependencies

* Linux (not tested on macOS, nor Windows)
* Python-3
* [OpenCV](https://opencv.org/)

## How to run it

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

https://www.youtube.com/watch?v=tILIeNjbH1E
http://thecolorsofmotion.com/films
https://digitalsynopsis.com/design/cinema-palettes-famous-movie-colors/

http://www.alanzucconi.com/2015/05/24/how-to-find-the-main-colours-in-an-image/
https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/

https://www.studiobinder.com/blog/how-to-use-color-in-film-50-examples-of-movie-color-palettes/



 convert df.png -color-matrix '  1.2 -0.1 -0.1
>                                 -0.1  1.2 -0.1
>                                 -0.1 -0.1  1.2 ' dd.png
