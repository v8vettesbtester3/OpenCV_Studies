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
    cv2.imshow("Canny Filtered",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
