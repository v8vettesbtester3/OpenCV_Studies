import cv2
import numpy as np
import utils


def enhanceEdges(src, dst, blurKsize=7, edgeKsize=5):
    '''
    Blur the image,
    convert to grayscale,
    find the edges,
    invert and rescale,
    combine edge image with original color image to enhance its edges
    '''

    if blurKsize >= 3:
        # kernel size is 3 or greater, therefore can do blurring operation
        # blur the image first to smooth out noise
        # otherwise speckles can be identified as having edges
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        # if the kernel size is 1, there is no meaning to blurring operation
        # so just convert from color to grayscale, without filtering first
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # use Laplacian filtering to emphasize edges in grayscale image
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)

    # An array for:
    # 1. normalizing (putting in the range of 0 - 1, and
    # 2. inverting (255 - pixel intensity)
    normalizedInverseAlpha = (1.0 / 255.0) * (255 - graySrc)

    # Convert original color image into three images:
    # a blue image, a green image and a red image
    # This permits handling each channel individually.
    channels = cv2.split(src)

    # Loop over individual color images and for each normalize and invert
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha

    # Recombine the blue, green and red components, putting the result in the image dst
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    ''' A filter that applies a convolution to V (or all of BGR) '''

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        # Apply the filter with a BGR or gray source/destination
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    ''' A sharpening filter with a 1-pixel radius
    Keeps the overall intensity of the image the same.
    Elements of kernel array sum to one.
    '''

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    ''' An edge-finding filter with a 1-pixel radius
    This makes edges white and non-edges black.
    Elements of kernel array sum to zero.
    '''

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    ''' A blur filter with a 2-pixel radius
    Keeps the overall intensity of the image the same.
    Elements of kernel array sum to one.'''

    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    ''' An emboss filter with a 1-pixel radius
    This blurs on one side and sharpens on the other,
    resulting in an embossed appearance.
    Keeps the overall intensity of the image the same.
    Elements of kernel array sum to one.
    '''

    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)


def cannyFilter(src):
    t_lower = 200
    t_upper = 300
    dst = cv2.Canny(src, t_lower, t_upper)
    dst = src
    # print(dst.shape)
    # for i in range(dst.shape[0]):   # 480
    #     for j in range (dst.shape[1]): # 640
    #         if dst[i][j] > 0:
    #             print("("+str(i)+","+str(j)+"): "+str(dst[i][j]))
    #print (dst)
    #input("Enter 2 continue")
    return dst
