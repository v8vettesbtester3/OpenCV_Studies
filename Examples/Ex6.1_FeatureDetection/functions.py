import cv2
import numpy
import os
from matplotlib import pyplot as plt

def update(sliderValue = 3):
    global gray
    global img

    val = cv2.getTrackbarPos('val','Harris Corners')
    if val % 2 == 0:
        val += 1
    if val > 31:
        val = 31
    if val < 3:
        val = 3

    img = cv2.imread('../../../images/5_of_diamonds.png') # read the static image file

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # make a grayscale image

    dst = cv2.cornerHarris(gray, 2, val, 0.04)

    img[dst > 0.01 * dst.max()] = [0, 255, 0]

    cv2.imshow("Harris Corners", img)



def detectHarrisCorners():
    # detect corners in a static image using the Harris method
    global gray
    global img

    cv2.namedWindow("Harris Corners")
    cv2.createTrackbar('val', 'Harris Corners', 3, 31, update)

    update()

    cv2.waitKey()

    cv2.destroyAllWindows()