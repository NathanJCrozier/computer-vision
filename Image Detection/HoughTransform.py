import numpy as np
import cv2 as cv


def HoughTransform(image): #Given an image, calculates the binary edge map and computes rho for a given number of angles in degrees. Returns the respective houghspace map.
    canny_image = cv.Canny(image, 0, 200) #Perform canny edge detection on our original image to prepare it for the hough transformation.
    width, height = canny_image.shape
    cv.imshow("canny", canny_image)
    max_rho = int(np.sqrt(width**2 + height**2)) #Find out the maximum possible value that rho can be in order to set the size of our houghspace
    max_theta = 180 #the maximum number of angles we are going to use
    houghspace_map = np.zeros((max_rho*2, max_theta)) #initialize our hough map (multiplying our rho-axis by 2 to look closer to the TA example)
    #Iterate over every pixel in our canny image looking for non-zero values
    for y in range(width):
        for x in range (height):
            if canny_image[y][x]==0:
                continue
            for t in range (max_theta): #when a non zero value is found, check it against every angle and find the different values for rho
                rho = round(x*np.cos(np.deg2rad(t)) + y*np.sin(np.deg2rad(t)))
                houghspace_map[rho][t] += 1 #increment the values at these indices

    return houghspace_map




def ShowHoughSpace(hough_map): #normalizes and cleans the map for a better viewing experience (easier to see the maxima). Also shows the map.
    hough_map_show = hough_map / hough_map.max() * 255
    hough_map_show = np.round(hough_map_show).astype(np.uint8)
    cv.imshow("Hough_Space", hough_map_show)



def GetHValues(h_map, threshhold): #Given a houghspace map and a threshold, finds the lines which meet the threshold and returns a list of them.
    width,height = h_map.shape
    h_list = []
    for y in range(height):
        for x in range(width):
            if h_map[x][y]==0: #If this cell got no votes, don't bother with it
                continue

            if h_map[x][y] > threshhold: #If this cell was above our threshold
                h_list.append((x,y)) #add it to our list of lines

    return h_list #return a list of ordered pairs containing our rho and theta values

def DrawHValues(image, h_list): #Given an image and a list of lines, this function draws these lines onto the image using cv.line
    for t in h_list: #for every ordered pair in our list
        rho = t[0] #get rho
        theta = t[1] #get theta

        #Convert back to cartesian coordinate form (method for conversion found in hough line transform tutorial in opencv documentation https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x_0 = a * rho
        y_0 = b * rho
        p1_x = int(x_0 + 1000 * (-b))
        p1_y = int(y_0 + 1000 * (a))
        p2_x = int(x_0 - 1000 * (-b))
        p2_y = int(y_0 - 1000 * (a))
        point_1 = (p1_x, p1_y)
        point_2 = (p2_x, p2_y)
        cv.line(image, point_1, point_2, (0, 0, 255)) #Draw a red line from point 1 to point 2

def DemoHough(choice): #For TAs and Demo purposes. This function takes an int of either 1 or 2 and demos Assignment 2 Part 1 on image 1 or 2 respectively.
    if choice == 1:
        original_image = np.asarray(cv.imread(cv.samples.findFile("hough1.png")))
        cv.imshow("Original", original_image)

        hough_map = HoughTransform(original_image)
        ShowHoughSpace(hough_map)

        h_maxima = GetHValues(hough_map, 20)
        DrawHValues(original_image, h_maxima)

        cv.imshow("Line", original_image)
        cv.waitKey(0)
        return

    else:

        if choice == 2:
            original_image = np.asarray(cv.imread(cv.samples.findFile("hough2.png")))
            cv.imshow("Original", original_image)

            hough_map = HoughTransform(original_image)
            ShowHoughSpace(hough_map)

            h_maxima = GetHValues(hough_map, 90)
            DrawHValues(original_image, h_maxima)

            cv.imshow("Line", original_image)
            cv.waitKey(0)
            return
        else:
            print("That wasn't an option")




#Uncomment this function and change the parameter to either 1 or 2 in order to demo Part 1 of Assignment 2

#DemoHough(1)
