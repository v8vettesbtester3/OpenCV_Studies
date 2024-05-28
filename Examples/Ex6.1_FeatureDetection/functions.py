import cv2
import numpy
import os
from matplotlib import pyplot as plt

def detectHarrisCorners():
    # detect corners in a static image using the Harris method

    img = cv2.imread('../../../images/5_of_diamonds.png') # read the static image file

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # make a grayscale image

    dst = cv2.cornerHarris(gray, 2, 23, 0.04)


    # Taking a matrix of size 5 as the kernel
    #kernel = numpy.ones((5, 5), numpy.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    #dst = cv2.dilate(dst, kernel, iterations = 1)


    img[dst > 0.01 * dst.max()] = [0, 255, 0]

    cv2.imshow("Harris corners", img)

    cv2.waitKey()

    cv2.destroyAllWindows()