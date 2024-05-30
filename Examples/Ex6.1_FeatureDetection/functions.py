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



def detectFASTFeatures():
    # detect corners in a static image using the FAST (Features from Accelerated Segment Test) method
    # Reference: https://docs.opencv.org/4.9.0/df/d0c/tutorial_py_fast.html
    # https://docs.opencv.org/4.x/df/d74/classcv_1_1FastFeatureDetector.html

    # Read a specifed image as its grayscale version
    img = cv2.imread('../../../images/5_of_diamonds.png', cv2.IMREAD_GRAYSCALE) # read the static image file as gray

    # Initialize a FAST object
    # Non-max suppression is used.
    fast = cv2.FastFeatureDetector.create(nonmaxSuppression=True)

    # Set the detection criteria
    # Detection of corners involves examining 16 pixels which
    # lie on a circle of radius 3 centered on the pixel in question.

    # A corner is identified to exist at a pixel if a certain number of surrounding
    # contiguous pixels are brighter than the central pixel by a specified threshold,
    # or, they are darker by that threshold.

    # There are 3 different types of detection available
    # 2: This is identified as TYPE_5_8, meaning that 5 contiguous pixels out of 8 satisfy the detection criteria.
    #    This is the least strict criterion.
    # 1: This is identified as TYPE_7_12, meaning that 7 contiguous pixels out of 12 satisfy the detection criteria.
    # 0: This is identified as TYPE_9_16, meaning that 9 contiguous pixels out of 16 satisfy the detection criteria.
    #    This is the most strict criterion.

    # Set criterion (type)
    stype = ''
    while stype not in ('5_8','7_12','9_16'):
        stype = input("Enter detection criterion (5_8, 7_12, 9_16): ")
    if stype == '5_8':
        type = 2
    elif stype == '7_12':
        type = 1
    else:
        type = 0
    fast.setType(type)

    # Set the threshold
    thresh = -1
    while thresh not in range(256):
        thresh = int(input("Enter threshold (0-255): "))
    fast.setThreshold(thresh)

    # Find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0))

    # Show the parameters
    print("Threshold: {}".format(fast.getThreshold()))
    print("Neighborhood: {}".format(fast.getType()))
    print("NonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("Total Keypoints: {}".format(len(kp)))

    cv2.imshow("FAST Corners", img2)


    cv2.waitKey()   # wait here until the image is closed.

    cv2.destroyAllWindows()


def orbFinder():
    # ORB (Oriented FAST and Rotated BRIEF)
    # FAST (Features from Accelerated Segment Test): keypoint detector
    # BRIEF (Binary Robust Independent Elementary Features): a descriptor

    # Load the images.
    img0 = cv2.imread('../../../images/nasa_logo.png', cv2.IMREAD_GRAYSCALE)    # query image
    img1 = cv2.imread('../../../images/kennedy_space_center.jpg', cv2.IMREAD_GRAYSCALE) # scene

    # Perform ORB feature detection and description
    orb = cv2.ORB.create()
    kp0, des0 = orb.detectAndCompute(img0, None)
    kp1, des1 = orb.detectAndCompute(img1, None)

    # Loop, trying different matching methods
    choice = -1
    while choice != 0:

        # Print menu
        while choice < 0 or choice > 3:
            print('Enter choice of matching method:')
            print('1: Use brute force matching method')
            print('2: Use k-nearest neighbors (knn) matching method')
            print('3: Use knn matching method with ratio testing')
            print('0: quit')
            schoice = input('')
            choice = int(schoice)

        if choice == 1:

            # Perform brute-force matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des0, des1)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw the best (up to) 25 matches
            img_matches = cv2.drawMatches(img0, kp0, img1, kp1, matches[:25], img1,
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            # Show the matches
            plt.imshow(img_matches)
            plt.show()

            choice = -1     # setup to repeat submenu

        elif choice == 2:

            # Perform knn matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            # return a list of lists, each such list no more than k matches
            pairs_of_matches = bf.knnMatch(des0, des1, k = 2)

            # Sort the pairs of matches by distance
            pairs_of_matches = sorted(pairs_of_matches, key=lambda x: x[0].distance)

            # Draw the best (up to) 25 matches, based on comparing the first element of each pair.
            img_pairs_of_matches = cv2.drawMatchesKnn(img0, kp0, img1, kp1, pairs_of_matches[:25], img1,
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            # Show the pairs of matches
            plt.imshow(img_pairs_of_matches)
            plt.show()

            choice = -1     # setup to repeat submenu

        elif choice == 3:

            # Perform knn matching with ratio filtering
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            # return a list of lists, each such list no more than k matches
            pairs_of_matches = bf.knnMatch(des0, des1, k = 2)

            # Sort the pairs of matches by distance
            pairs_of_matches = sorted(pairs_of_matches, key=lambda x: x[0].distance)

            # Apply the ratio test
            # Create a list of the first elements of the pairs of matches, where
            # (a) there is a pair and
            # (b) the distance (smaller is closer match) of the first element is
            #     less than 80% of the distance of the second element.
            matches = [x[0] for x in pairs_of_matches
                       if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]

            # Draw the best (up to) 25 matches
            img_matches = cv2.drawMatches(img0, kp0, img1, kp1, matches[:25], img1,
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            # Show the matches
            plt.imshow(img_matches)
            plt.show()

            choice = -1     # setup to repeat submenu

        else:
            break   # leave submenu loop



def orbFinderFLANN():

    # Load the images.
    img0 = cv2.imread('../../../images/nasa_logo.png', cv2.IMREAD_GRAYSCALE)    # query image
    img1 = cv2.imread('../../../images/kennedy_space_center.jpg', cv2.IMREAD_GRAYSCALE) # scene

    # Perform ORB feature detection and description
    # Reference: https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
    orb = cv2.ORB.create()

    # Set the scale factor
    scale = -1
    while scale < 1.0 or scale > 2.0:
        scale = float(input("Enter scale factor (1.0 - 2.0): "))
    orb.setScaleFactor(scale)

    # Set number of levels
    size = img1.shape
    minDimension = min(size[0], size[1])
    nlevels = int(numpy.log(minDimension/4) / numpy.log(scale))
    print("Number of levels: ", nlevels)
    orb.setNLevels(nlevels)

    # Set the threshold
    thresh = -1
    while thresh not in range(256):
        thresh = int(input("Enter threshold (0-255): "))
    orb.setFastThreshold(thresh)

    kp0, des0 = orb.detectAndCompute(img0, None)
    kp1, des1 = orb.detectAndCompute(img1, None)

    # Define FLANN-based matching parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    # Perform FLANN-based matching
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    des0 = numpy.float32(des0)
    des1 = numpy.float32(des1)
    matches = matcher.knnMatch(des0, des1, k = 2)

    # Prepare an empty mask to draw good matches
    mask_matches = [[0,0] for i in range(len(matches))]

    # Populate the mask based on the ratio test
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:   # originally, threshold was 0.7
            mask_matches[i] = [1, 0]
            #print("Match at i =",i)

    # Draw the matches that passed the ratio test
    draw_params = dict(matchColor=(0, 255, 0),          # color tuple in (R,G,B) order
                       singlePointColor=(255, 0, 0),    # color tuple in (R,G,B) order
                       matchesMask=mask_matches,
                       flags=0)
    img_matches = cv2.drawMatchesKnn(img0, kp0, img1, kp1, matches, None, **draw_params)

    # Show the matches
    plt.imshow(img_matches)
    plt.show()

    cv2.waitKey()   # wait here until the image is closed.
