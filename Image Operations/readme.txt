Written by: Nathan Crozier
SID: 40048644
------------------------------------------------------------------------------------------------------------------
Below is a list of my functions and a brief description of what they do. However, I have attempted to streamline 
the process by including three extra functions at the bottom which are specifically for the purposes of demonstrating 
each part in full. They are ready to be run at the bottom of the document, and are currently commented out. 
Feel free to un-comment one and then run it to see the results. You can then re-comment it and try another one.
The image used was the one provided in the assignment from Wikiart. 
I have included it in the zip in case you need it under the name "sample.jpg".
-------------------------------------------------------------------------------------------------------------------
downSampleImage(int, image): downsamples an image by a given factor

downSampleDemo(image): run the downsample function for a preset series of numbers and display the results

upSampleNN(image): upsamples an image using nearest neighbour interpolation

upSampleBilinear(image): upsamples an image using bilinear interpolation

upSampleBicubic(image): upsamples an image using bicubic interpolation

diagonalShift(shift_amt, image): shifts a given image diagonally to the right by a given amount.

createGaussianFilter(size, sigma): creates a gaussian kernel of given size and deviation

applyGaussian(size, sigma, image): applies a gaussian filter to an image

gaussianDifference(a, b, sigma_a, sigma_b, image): creates and applies two different gaussian kernels to an image, and displays the difference.

sobelX(image): runs the sobel operator over an image w.r.t x

sobelY(image):runs the sobel operator over an image w.r.t y

orientation(sbx_image, sby_image): returns the orientation of an image when given the sobel x and y images from earlier.
   
magnitude(sbx_image, sby_image): returns the magnitude of an image when given the sobel x and y images from earlier.
 
partOneDemo(image): (IMPORTANT). This performs all of the requisite code for part 1 (un-comment it out at the bottom to run it!)

partTwoDemo(image): (IMPORTANT). This performs all of the requisite code for part 2 (un-comment it out at the bottom to run it!)

partThreeDemo(image): (IMPORTANT). This performs all of the requisite code for part 3 (un-comment it out at the bottom to run it!)
