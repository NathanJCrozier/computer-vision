# Computer Vision
A collection of projects completed while learning computer vision.

## Image Operations
This project was broken down into three parts. 
1. Sampling/Interpolation
2. Filtering
3. Edge detection

In part 1 I had to write a function that could take a color image with three color channels (RGB) and downsample it by a factor of 2 to 16. I then had to upsample this image back up using multiple interpolation techniques (i.e. nn, bilinear, bicubic).

In part 2 I had to write various filters which would manipulate an image. One filter shifts an image diagonally towards the top right corner. Another filter calculates the the sum of two second partial order derivatives with respect to x and y, effectively creating a gaussian filter which can be applied to an image. I was then required to compute two gaussian filters (scaled by different parameters) and calculate the different of gaussians.

In part 3 I was tasked with writing functions to apply the Sobel operators with respect to x and y. I then had to write a function to calculate the orientation of each pixel using the sobel x and y values for the image. Lastly I used those same values in another function which I used to calculate the magnitude of each pixel in the image.

## Image Detection
This project was broken down into three parts as well.
