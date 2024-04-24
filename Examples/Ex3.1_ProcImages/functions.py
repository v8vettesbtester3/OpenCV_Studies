import cv2
import numpy as np
import os
from scipy import ndimage

def highPassFiltering():
    kernel_3x3 = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1,-1,-1]])

    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1,1,2,1,-1],
                           [-1, 2,-4,2,-1],
                           [-1,1,2,1,-1],
                           [-1, -1, -1, -1, -1]])

    img = cv2.imread("../../../images/statue_small.jpg", 0)

    k3 = ndimage.convolve(img, kernel_3x3)
    k5 = ndimage.convolve(img, kernel_5x5)

    blurred = cv2.GaussianBlur(img, (17, 17), 0)

    g_hpf = img - blurred

    cv2.imshow("3x3", k3)
    cv2.imshow("5x5", k5)
    cv2.imshow("blurred", blurred)
    cv2.imshow("g_hpf",g_hpf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cannyFiltering():
    img = cv2.imread("../../../images/statue_small.jpg", 0)
    t_lower = int(input("Enter lower threshold: "))
    t_upper = int(input("Enter upper threshold: "))
    dst = cv2.Canny(img, t_lower, t_upper)
    cv2.imshow("Original",img)
    cv2.imshow("Canny Filtered",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour1():
    img = np.zeros((200,200), dtype=np.uint8)
    img[50:150, 50:150] = 255

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
    cv2.imshow("contours", color)
    cv2.waitKey()
    cv2.destroyAllWindows()

def contour2():
    # Read a sample image file, without changing it.  Adjust your path to match your location of the images folder.
    # Then, down-sample the input image using pyrDown()
    # Reference: https://docs.opencv.org/4.x/d4/d1f/tutorial_pyramids.html
    img = cv2.imread("../../../images/hammer.jpg", cv2.IMREAD_UNCHANGED)
    cv2.imshow("Source: Full Size", img)
    img = cv2.pyrDown(img)
    cv2.imshow("Source: Half Size", img)


    # Convert the color image to a grayscale image, then threshold.
    # Thresholding in THRESH_BINARY mode: if source pixel intensity is > thresh (127), then it is set to maxval (255)
    # Return values: ret is the threshold used (127) and thresh is the thresholded image.
    # Reference: https://docs.opencv.org/4.9.0/d7/d4d/tutorial_py_thresholding.html
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholded", thresh)

    # Find the contours in a binary image
    # Return values: contours is a list of the contours found,
    # hier[0] is a 2D array
    # hier[0][i] are the 4 hierarchical elements of contour i
    # hier[0][i][0] is the index of the next contour
    # hier[0][i][1] is the index of the previous contour
    # hier[0][i][2] is the index of the first child contour
    # hier[0][i][3] is the index of the parent contour
    # if any hier[0][i][j] is negative, the corresponding contour does not exist
    contours, hier = cv2.findContours(thresh,                   # source image
                                      cv2.RETR_EXTERNAL,        # contour retrieval mode
                                      cv2.CHAIN_APPROX_SIMPLE)  # contour approximation method

    for c in contours:

        # find bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

        # find the minimum area
        rect = cv2.minAreaRect(c)
        # calculate coordinates of hte minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.intp(box)
        # draw contours
        cv2.drawContours(img, [box], 0, (0,0,255), 0)

        # calculate the center and radius of minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # cast to integers
        center = (int(x), int(y))
        radius = int(radius)
        # draw the circle
        img = cv2.circle(img, center, radius, (0,255,0), 2)

    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    cv2.imshow("contours", img)

    cv2.imwrite('h.jpg', img)

    cv2.waitKey()
    cv2.destroyAllWindows()
