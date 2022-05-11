Written by: Nathan Crozier
------------------------------------------------------------------------------------------------------------------
Below is a list of my functions and a brief description of what they do. However, I have attempted to streamline 
the process by including extra functions at the bottom which are specifically for the purposes of demonstrating 
each part in full. They are ready to be run at the bottom of the document, and are currently commented out. 
Feel free to un-comment one and then run it to see the results. You can then re-comment it and try another one.
The images used were the ones provided in the assignment. 
I have included them in the zip in case you need them.
-------------------------------------------------------------------------------------------------------------------

--------------------------------------
Functions in HoughTransform.py (Part 1)
--------------------------------------
def HoughTransform(image): #Given an image, calculates the binary edge map and computes rho for a given number of angles in degrees. Returns the respective houghspace map.
 
def ShowHoughSpace(hough_map): #normalizes and cleans the map for a better viewing experience (easier to see the maxima). Also shows the map.

def GetHValues(h_map, threshhold): #Given a houghspace map and a threshold, finds the lines which meet the threshold and returns a list of them.

def DrawHValues(image, h_list): #Given an image and a list of lines, this function draws these lines onto the image using cv.line

def DemoHough(choice): #For TAs and Demo purposes. This function takes an int of either 1 or 2 and demos Assignment 2 Part 1 on image 1 or 2 respectively.

--------------------------------------
Functions in HarrisFeatures.py (Part 2 and 3)
--------------------------------------

def Orientation(sbx_image, sby_image): #returns the orientation of an image when given the sobel x and y images

def Magnitude(sbx_image, sby_image): #returns the magnitude of an image when given the sobel x and y images

def GetLocalMaxima(r_matrix, threshold): #Given a corner response matrix, performs non maximum suppression and thresholds the values.

def HarrisCornerDetect(image, k, window_size = 5 ): #Compute the corner response in a window around each pixel, returns it as a matrix

def NormalizeToShow(r_matrix): #Normalizes the r_matrix for display purposes

def GetKeyPoints(corner_list): #By-point conversion of ordered pairs into KeyPoint class objects, returns a list of these objects

def CleanKeyPointsForSIFT(key_points, image): #Removes boundary key points which cannot have a proper SIFT descriptor calculated

def DemoHarris(image, threshold): #For TA and Demo purposes. Takes an image and demos Assignment 2 Part 2 on it using the functions shown above

def GetDescriptors(image, key_points): #Given an image and a list of key points found earlier, computes descriptors for each key point and returns a descriptor list in the same order.

def DemoMatching(image1, image2, detect_threshold, match_threshold): #For TA and Demo purposes. Takes both images and their thresholds and uses them to demo Assignment 2 Part 3 (Warning: takes a little bit of time!)
