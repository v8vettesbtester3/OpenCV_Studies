import cv2
import numpy
import os
from matplotlib import pyplot as plt

img = None
gray = None
blockSize = 2
sobelAperture = 23
freeParam = 40
detectThresh = 10


def readImage():
    # Read a specifed image and calculate its grayscale version
    global gray, img

    img = cv2.imread('../../../images/5_of_diamonds.png') # read the static image file

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # make a grayscale image


def calcCorners():
    # Use the Harris method to identify corners.
    global gray, img, blockSize, sobelAperture, freeParam, detectThresh

    k = 0.001*freeParam # larger values: more precise, but may miss corners
    # Reference: https://stackoverflow.com/questions/54720646/what-does-ksize-and-k-mean-in-cornerharris
    dst = cv2.cornerHarris(gray, blockSize, sobelAperture, k)   # Harris method

    # Make every pixel whose Harris value is greater than threshold shown as green
    img[dst > 0.001 * detectThresh * dst.max()] = [0, 255, 0]


def processHarrisCorners():
    # Common compound operation: read the image, calculate its corners and display it.
    global img

    readImage()

    calcCorners()

    cv2.imshow("Harris Corners", img)



def updateBlockSize(sliderPosition = 0):
    # Event handler for moving Block Size trackbar position.

    global blockSize

    # Get the position of the Block Size value trackbar
    blockSize = cv2.getTrackbarPos('blockSize','Harris Corners')

    processHarrisCorners()  # calculate and show the result


def updateSobelAperture(sliderPosition = 0):
    # Event handler for moving Sobel Aperture trackbar position.

    global sobelAperture

    # Get the position of the Sobel Aperture value trackbar
    sobelAperture = cv2.getTrackbarPos('sobelAperture','Harris Corners')
    if sobelAperture % 2 == 0:
        sobelAperture += 1      # Sobel aperture must be an odd number
    if sobelAperture > 31:
        sobelAperture = 31      # Sobel aperture can be no larger than 31
    if sobelAperture < 3:
        sobelAperture = 3       # Sobel aperture can be no smaller than 3

    processHarrisCorners()  # calculate and show the result


def updateFreeParam(sliderPosition = 0):
    # Event handler for moving Free Parameter trackbar position.

    global freeParam

    # Get the position of the Free Parameter value trackbar
    freeParam = cv2.getTrackbarPos('freeParam','Harris Corners')

    processHarrisCorners()  # calculate and show the result


def updateDetectionThresh(sliderPosition = 0):
    # Event handler for moving Detection Threshold trackbar position.

    global detectThresh

    # Get the position of the Detection Threshold value trackbar
    detectThresh = cv2.getTrackbarPos('det Thresh','Harris Corners')

    processHarrisCorners()  # calculate and show the result




def detectHarrisCorners():
    # detect corners in a static image using the Harris method
    # Reference: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html

    global blockSize, sobelAperture, freeParam, detectThresh

    cv2.namedWindow("Harris Corners")

    # Establish a track bar with a maximum value of 10, starting out at position 2.
    cv2.createTrackbar('blockSize', 'Harris Corners', blockSize, 10, updateBlockSize)
    cv2.setTrackbarMin('blockSize', 'Harris Corners', 1)    # minimum value of track bar = 1.

    # Establish a track bar with a maximum value of 31, starting out at position 23.
    cv2.createTrackbar('sobelAperture', 'Harris Corners', sobelAperture, 31, updateSobelAperture)
    cv2.setTrackbarMin('sobelAperture', 'Harris Corners', 3)    # minimum value of track bar = 3.

    # Establish a track bar with a maximum value of 60, starting out at position 40.
    cv2.createTrackbar('freeParam', 'Harris Corners', freeParam, 60, updateFreeParam)
    cv2.setTrackbarMin('freeParam', 'Harris Corners', 40)    # minimum value of track bar = 40.

    # Establish a track bar with a maximum value of 100, starting out at position 10.
    cv2.createTrackbar('det Thresh', 'Harris Corners', detectThresh, 100, updateDetectionThresh)
    cv2.setTrackbarMin('det Thresh', 'Harris Corners', 0)    # minimum value of track bar = 1.

    # Initialize the trackbars
    updateBlockSize()
    updateSobelAperture()
    updateFreeParam()
    updateDetectionThresh()

    cv2.waitKey()   # wait here until the image is closed.

    cv2.destroyAllWindows()
