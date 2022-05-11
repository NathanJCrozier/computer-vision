import numpy as np
import cv2 as cv


def Orientation(sbx_image, sby_image): #returns the orientation of an image when given the sobel x and y images
    row, col = np.asarray(sbx_image).shape
    ori_image = np.zeros((row, col))
    for r in range(0, row):
        for c in range(0, col):
            if sbx_image[r,c] == 0:
                ori_image[r,c] = 0
            else:
                ori_image[r, c] = np.arctan(sby_image[r, c]/sbx_image[r, c])
    return ori_image


def Magnitude(sbx_image, sby_image): #returns the magnitude of an image when given the sobel x and y images
    row,col = np.asarray(sbx_image).shape
    mag_image = np.zeros((row, col))
    for r in range(0, row):
        for c in range (0, col):
            mag_image[r, c] = np.sqrt(sbx_image[r, c]**2 + sby_image[r, c]**2)
    return mag_image

def GetLocalMaxima(r_matrix, threshold): #Given a corner response matrix, performs non maximum suppression and thresholds the values.
    corner_list = []
    height, width = r_matrix.shape
    local_max = cv.dilate(r_matrix, np.ones((3, 3)))
    for y in range(height):
        for x in range(width):
            if local_max[y][x] > threshold and local_max[y][x] == r_matrix[y][x]:
                corner_list.append([x, y, local_max[y][x]])

    corner_list.sort(key=lambda corner: corner[2], reverse=True)

    return corner_list

def HarrisCornerDetect(image, k, window_size = 5 ): #Compute the corner response in a window around each pixel, returns it as a matrix
    #Compute the gradients Ix, Iy
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    i_x = np.gradient(gray_image)[0]
    i_y = np.gradient(gray_image)[1]


    #Compute Ix^2, Iy^2, IxIy and smooth with a Gaussian
    i_xx = i_x**2
    i_yy = i_y**2
    i_xy = i_x * i_y

    i_xx = cv.GaussianBlur(i_xx, (3,3), 3)
    i_yy = cv.GaussianBlur(i_yy, (3,3), 3)
    i_xyy = cv.GaussianBlur(i_xy, (3,3), 3)

    #Part 2 B: Displaying the results of the gradients
    cv.imshow("Ix", i_x)
    cv.imshow("Iy", i_y)
    cv.imshow("Ixy", i_xy)

    #Compute the Harris matrix H in a window around each pixel
    offset = int(window_size/2)

    height, width = image.shape[0:2]
    r_matrix = np.zeros((height, width))

    #Using the sliding window technique and sum of squares to calculate the Harrix matrix
    #Technique adapted from that of an ACM script provided by Aditya Intwala
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window_i_xx = i_xx[y - offset: y + offset + 1, x - offset: x + offset + 1]
            window_i_yy = i_yy[y - offset: y + offset + 1, x - offset: x + offset + 1]
            window_i_xy = i_xy[y - offset: y + offset + 1, x - offset: x + offset + 1]

            s_xx = window_i_xx.sum()
            s_yy = window_i_yy.sum()
            s_xy = window_i_xy.sum()

            #Compute corner response function R

            det = (s_xx * s_yy) - (s_xy**2)
            trace = (s_xx + s_yy)
            r = det - k*(trace**2)
            r_matrix[y][x] = r

    return r_matrix

def NormalizeToShow(r_matrix): #Normalizes the r_matrix for display purposes
    r_matrix_to_show = r_matrix / np.max(r_matrix)
    return r_matrix_to_show

def GetKeyPoints(corner_list): #By-point conversion of ordered pairs into KeyPoint class objects, returns a list of these objects
    key_points = []

    for point in corner_list:
        key_points.append(cv.KeyPoint(point[0], point[1], 1))

    return key_points

def CleanKeyPointsForSIFT(key_points, image): #Removes boundary key points which cannot have a proper SIFT descriptor calculated
    found_badpoint = True
    while(found_badpoint == True):
        found_badpoint = False
        length = len(key_points)
        for i in range (length):
            if (key_points[i].pt[1] - 8) < 0 or (key_points[i].pt[1] + 8) > image.shape[0] or (key_points[i].pt[0] - 8) < 0 or (key_points[i].pt[0] + 8) > image.shape[1]:
                key_points.pop(i)
                found_badpoint = True
                break

    return key_points



def DemoHarris(image, threshold): #For TA and Demo purposes. Takes an image and demos Assignment 2 Part 2 on it using the functions shown above
    r_matrix = HarrisCornerDetect(image, 0.04, 5)

    # Threshold R and find local maxima of response function (non-maximum suppression)
    corner_list = GetLocalMaxima(r_matrix, threshold)
    key_points = GetKeyPoints(corner_list)

    corner_image = image.copy()
    #Part 2 D1: Drawing key points
    cv.drawKeypoints(image, key_points, corner_image)

    # Part 2 C: Displaying the results of response (R)
    pretty_response = NormalizeToShow(r_matrix)
    cv.imshow("Corner Response", pretty_response)

    #Part 2 D2: Displaying the final interest points
    cv.imshow("Corner", corner_image)

    cv.waitKey(0)



def GetDescriptors(image, key_points): #Given an image and a list of key points found earlier, computes descriptors for each key point and returns a descriptor list in the same order.
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    i_x = np.gradient(gray_image)[0]
    i_y = np.gradient(gray_image)[1]

    ori_image = Orientation(i_x, i_y)
    mag_image = Magnitude(i_x, i_y)
    knum = 0
    dnum = 0
    descriptors = []

    key_points = CleanKeyPointsForSIFT(key_points, image)

    for kpt in key_points:
        knum += 1
        x_coor = int(kpt.pt[0])
        y_coor = int(kpt.pt[1])
        startx = x_coor - 8
        endx = x_coor + 8
        starty = y_coor - 8
        endy = y_coor + 8


        descriptor = []

        bin0 = 0
        bin45 = 0
        bin90 = 0
        bin135 = 0
        bin180 = 0
        bin225 = 0
        bin270 = 0
        bin315 = 0


        #Split our orientation and magnitude gradients into 16x16 windows
        ori_win = ori_image[starty:endy, startx : endx]
        mag_win = mag_image[starty:endy, startx : endx]
        #Split our windows into 8x8 grids of cells
        ori_tcells = ori_win[0:8 , 0:16]
        ori_bcells = ori_win[8:16, 0:16]
        ori_topl_cell = ori_tcells[0:8, 0:8]
        ori_topr_cell = ori_tcells[0:8, 8:16]
        ori_botl_cell = ori_bcells[0:8, 0:8]
        ori_botr_cell = ori_bcells[0:8, 8:16]

        ori_bigcells = [ori_topl_cell, ori_topr_cell, ori_botl_cell, ori_botr_cell]


        mag_tcells = mag_win[0:8 , 0:16]
        mag_bcells = mag_win[8:16, 0:16]
        mag_topl_cell = mag_tcells[0:8, 0:8]
        mag_topr_cell = mag_tcells[0:8, 8:16]
        mag_botl_cell = mag_bcells[0:8, 0:8]
        mag_botr_cell = mag_bcells[0:8, 8:16]

        mag_bigcells = [mag_topl_cell, mag_topr_cell, mag_botl_cell, mag_botr_cell]
        #Split our 8x8 cells into 4x4 subcells
        for i in range (4):

            ori_sub_cell0 = ori_bigcells[i][0:4, 0:4]
            ori_sub_cell1 = ori_bigcells[i][0:4, 4:8]
            ori_sub_cell2 = ori_bigcells[i][4:8, 0:4]
            ori_sub_cell3 = ori_bigcells[i][4:8, 4:8]

            mag_sub_cell0 = mag_bigcells[i][0:4, 0:4]
            mag_sub_cell1 = mag_bigcells[i][0:4, 4:8]
            mag_sub_cell2 = mag_bigcells[i][4:8, 0:4]
            mag_sub_cell3 = mag_bigcells[i][4:8, 4:8]

            ori_smallcells = [ori_sub_cell0, ori_sub_cell1, ori_sub_cell2, ori_sub_cell3]
            mag_smallcells = [mag_sub_cell0, mag_sub_cell1, mag_sub_cell2, mag_sub_cell3]

            #Compute a histogram for each subcell with 8 bins per histogram
            for c in range(4):
                orientations = ori_smallcells[c]
                magnitudes = mag_smallcells[c]
                for y in range(4):
                    for x in range(4):
                        theta = np.rad2deg(orientations[y][x])
                        if theta < 0:
                            theta += 360
                        if theta < 45:
                            bin0 += magnitudes[y][x]
                        elif theta < 90:
                            bin45 += magnitudes[y][x]
                        elif theta < 135:
                            bin90 += magnitudes[y][x]
                        elif theta < 180:
                            bin135 += magnitudes[y][x]
                        elif theta < 225:
                            bin180 += magnitudes[y][x]
                        elif theta < 270:
                            bin225 += magnitudes[y][x]
                        elif theta < 315:
                            bin270 += magnitudes[y][x]
                        elif theta < 360:
                            bin315 += magnitudes[y][x]
                descriptor.extend([bin0, bin45, bin90, bin135, bin180, bin225, bin270, bin315]) #Flush the bins to our global descriptor list and clear them
                bin0 = 0
                bin45 = 0
                bin90 = 0
                bin135 = 0
                bin180 = 0
                bin225 = 0
                bin270 = 0
                bin315 = 0
                bin360 = 0
        dnum += 1
        print("Adding descriptor " + str(dnum) + " for keypoint " + str(knum))
        descriptors.append(descriptor)
    return descriptors


def DemoMatching(image1, image2, detect_threshold, match_threshold): #For TA and Demo purposes. Takes both images and their thresholds and uses them to demo Assignment 2 Part 3 (Warning: takes a little bit of time!)
    r_matrix_1 = HarrisCornerDetect(image1, 0.04, 5)
    corner_list_1 = GetLocalMaxima(r_matrix_1, detect_threshold)
    key_points_1 = GetKeyPoints(corner_list_1)
    descriptors_1 = GetDescriptors(image1, key_points_1)

    r_matrix_2 = HarrisCornerDetect(image2, 0.04, 5)
    corner_list_2 = GetLocalMaxima(r_matrix_2, detect_threshold)
    key_points_2 = GetKeyPoints(corner_list_2)
    descriptors_2 = GetDescriptors(image2, key_points_2)

    squared_differences = []

    matches = []

    SSD = 0
    #Performing the ratio test to match key points using their descriptors
    for i in range (len(descriptors_1)):
        for j in range(len(descriptors_2)):
            d1 = descriptors_1[i]
            d2 = descriptors_2[j]
            for k in range(128):
                square_diff = (d1[k] - d2[k])**2
                squared_differences.append(square_diff)
            for value in squared_differences:
                SSD += value
            if SSD < match_threshold:
                matches.append(cv.DMatch(i, j, SSD))
            squared_differences.clear()
            SSD = 0

    print(matches)
    output = np.zeros((image1.shape[0], image1.shape[1]))
    output = cv.drawMatches(image1, key_points_1, image2, key_points_2, matches, output)
    cv.imshow("matches", output)
    cv.waitKey(0)






image1 = np.asarray(cv.imread(cv.samples.findFile("Yosemite1.jpg")))
image2 = np.asarray(cv.imread(cv.samples.findFile("Yosemite2.jpg")))



#Uncomment the functions below in order to quickly test the code.

#DemoHarris(image1, 350000000)
#DemoHarris(image2, 250000000)
#DemoMatching(image1, image2, 100000000, 99999) #KeyPoint threshold has been lowered to allow for more matches

