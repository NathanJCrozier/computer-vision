import numpy as np
import cv2 as cv

def downSampleImage(int, image):
    original_image = np.asarray(image) #import our given image as an array
    new_image = original_image[1::int, 1::int] #start at index 1 and take every nth row/column as our new image

    return new_image

def downSampleDemo(image): #run the downsample function for a preset series of numbers and display the results
    og_image = np.asarray(image)
    image1 = downSampleImage(2, og_image)
    image2 = downSampleImage(4, og_image)
    image3 = downSampleImage(8, og_image)
    image4 = downSampleImage(16, og_image)
    cv.imshow("Original Image", og_image)
    cv.imshow("Downsampled Image [2]", image1)
    cv.imshow("Downsampled Image [4]", image2)
    cv.imshow("Downsampled Image [8]", image3)
    cv.imshow("Downsampled Image [16]", image4)


def upSampleNN(image):
    original_image = np.asarray(image)
    new_image = cv.resize(original_image, None,  fx=10, fy=10, interpolation=cv.INTER_NEAREST)
    return new_image

def upSampleBilinear(image):
    original_image = np.asarray(image)
    new_image = cv.resize(original_image, None,  fx=10, fy=10, interpolation=cv.INTER_LINEAR)
    return new_image

def upSampleBicubic(image):
    original_image = np.asarray(image)
    new_image = cv.resize(original_image, None,  fx=10, fy=10, interpolation=cv.INTER_CUBIC)
    return new_image

def diagonalShift(shift_amt, image): #shifts a given image diagonally to the right by a given amount.
    original_image = np.asarray(image)
    num_columns = original_image.shape.__getitem__(1)
    num_rows = original_image.shape.__getitem__(0)
    new_image = np.copy(original_image)

    for i in range(num_rows):
        for j in range(num_columns):
            if i >= num_rows - shift_amt or j <= shift_amt:
                new_image[i,j] = 255 #if the pixel we would have used to replace this one doesn't exist, then replace it with a white pixel.
            else: new_image[i,j] = original_image[i+shift_amt, j-shift_amt]

    return new_image


def createGaussianFilter(size, sigma): #creates a gaussian kernel of given size and deviation
    N = size
    A = sigma * np.sqrt(2*np.pi)
    l_term = 1/A

    half_point = int(N / 2)

    if half_point % 2 == 0: #we can't have a proper middle point if we get an even number.. our kernel needs to be odd.
        half_point += 1
        N = int(half_point * 2)

    x_c = half_point
    y_c = half_point

    kernel = np.zeros((N, N))

    for r in range(0, N):
        for c in range (0, N):
            x_value = float((c-x_c)**2)
            y_value = float((r-y_c)**2)
            kernel[c, r] = l_term * np.exp(-(x_value + y_value)/(2.0*sigma**2))


    norm_kernel = kernel/np.sum(kernel) #normalize our kernel so that it sums up to equal 1.
    return norm_kernel

def applyGaussian(size, sigma, image): #applies a gaussian filter to an image
    kernel = createGaussianFilter(size, sigma)
    new_image = cv.filter2D(image, -1, kernel)

    return new_image

def gaussianDifference(a, b, sigma_a, sigma_b, image): #creates and applies two different gaussian kernels to an image, and displays the difference.
    first_kernel = createGaussianFilter(a, sigma_a)
    second_kernel = createGaussianFilter(b, sigma_b)
    first_image = cv.filter2D(image, -1, first_kernel)
    second_image = cv.filter2D(image, -1, second_kernel)
    dif_image = cv.absdiff(first_image, second_image)
    cv.imshow("Original", image)
    cv.imshow("Gaussian: sigma = [5]", first_image)
    cv.imshow("Gaussian: sigma = [3]", second_image)
    cv.imshow("Difference of Gaussians", dif_image)
    return dif_image

def sobelX(image): #runs the sobel operator over an image w.r.t x
    sob_kernel = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    new_image = cv.filter2D(image, -1, sob_kernel)
    return new_image


def sobelY(image):#runs the sobel operator over an image w.r.t y
    sob_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    new_image = cv.filter2D(image, -1, sob_kernel)
    return new_image

def orientation(sbx_image, sby_image): #returns the orientation of an image when given the sobel x and y images from earlier.
    row, col = np.asarray(sbx_image).shape
    ori_image = np.zeros((row, col))
    for r in range(0, row):
        for c in range(0, col):
            ori_image[r, c] = np.arctan(sby_image[r, c]/sbx_image[r, c])
    return ori_image


def magnitude(sbx_image, sby_image): #returns the magnitude of an image when given the sobel x and y images from earlier.
    row,col = np.asarray(sbx_image).shape
    mag_image = np.zeros((row, col))
    for r in range(0, row):
        for c in range (0, col):
            mag_image[r, c] = np.sqrt(sbx_image[r, c]**2 + sby_image[r, c]**2)
    return mag_image

def partOneDemo(image): #For the marker. This performs all of the requisite code for part 1 (un-comment it out at the bottom to run it!)
    original_image = np.asarray(image)
    downSampleDemo(original_image)
    di_sixteen = downSampleImage(16, original_image)
    ups_nn = upSampleNN(di_sixteen)
    ups_bili = upSampleBilinear(di_sixteen)
    ups_bicu = upSampleBicubic(di_sixteen)
    cv.imshow("Upsampled using nearest neighbour", ups_nn)
    cv.imshow("Upsampled using bilinear", ups_bili)
    cv.imshow("Upsampled using bicubic", ups_bicu)
    cv.waitKey(0)
    return 0

def partTwoDemo(image): #For the marker. This performs all of the requisite code for part 2 (un-comment it out at the bottom to run it!)
    original_image = np.asarray(image)
    diag_image = diagonalShift(100, original_image)
    cv.imshow("Diagonally Shifted Image", diag_image)
    g_image = applyGaussian(5, 3, original_image)
    gaussianDifference(9, 3, 5, 3, original_image)
    return 0

def partThreeDemo(image): #For the marker. This performs all of the requisite code for part 3 (un-comment it out at the bottom to run it!)
    original_image = np.asarray(image)
    sobx_image = sobelX(original_image)
    soby_image = sobelY(original_image)
    edge_image = cv.Canny(original_image, 100, 300)
    cv.imshow("Sobel w.r.t x", sobx_image)
    cv.imshow("Sobel w.r.t y", soby_image)
    orientation_image = orientation(sobx_image, soby_image)
    cv.imshow("Orientation of sample image", orientation_image)
    magnitude_image = magnitude(sobx_image, soby_image)
    cv.imshow("Magnitude of sample image", magnitude_image)
    cv.imshow("Canny Edge Map", edge_image)

    return 0

sample_image_grayscale = np.asarray(cv.imread(cv.samples.findFile("sample.jpg"), 0)) #grabbing the sample image in grayscale
sample_image_color = np.asarray(cv.imread(cv.samples.findFile("sample.jpg"))) #grabbing the sample image in color
#partOneDemo(sample_image_color)
#partTwoDemo(sample_image_grayscale)
#partThreeDemo(sample_image_grayscale)
cv.waitKey(0)