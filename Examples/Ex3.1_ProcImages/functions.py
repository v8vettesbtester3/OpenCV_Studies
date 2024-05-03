import cv2
import numpy as np
from scipy import ndimage


def highPassFiltering():
    kernel_3x3 = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, 2, -4, 2, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, -1, -1, -1, -1]])

    img = cv2.imread("../../../images/statue_small.jpg", 0)    # set path correctly for your installation.

    k3 = ndimage.convolve(img, kernel_3x3)
    k5 = ndimage.convolve(img, kernel_5x5)

    blurred = cv2.GaussianBlur(img, (17, 17), 0)

    g_hpf = img - blurred

    cv2.imshow("3x3", k3)
    cv2.imshow("5x5", k5)
    cv2.imshow("blurred", blurred)
    cv2.imshow("g_hpf", g_hpf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cannyFiltering():
    # Apply Canny filtering to obtain edges
    # If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge
    # If a pixel gradient value is below the lower threshold, then it is rejected.
    # If the pixel gradient is between the two thresholds, then it will be accepted only if
    # it is connected to a pixel that is above the upper threshold.

    img = cv2.imread("../../../images/statue_small.jpg", 0)    # set path correctly for your installation.
    t_lower = int(input("Enter lower threshold: "))
    t_upper = int(input("Enter upper threshold: "))
    dst = cv2.Canny(img, t_lower, t_upper)
    cv2.imshow("Original", img)
    cv2.imshow("Canny Filtered", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour1():
    img = np.zeros((200, 200), dtype=np.uint8)
    img[50:150, 50:150] = 255

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    perimeter = cv2.arcLength(contours[0], True)
    print("100 x 100 square.  Perimeter =",perimeter)

    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contours", color)
    cv2.waitKey()
    cv2.destroyAllWindows()


def contour2():
    # Exhibit contour detection and the determination of various bounding shapes.

    # Generates 7 images:
    # 1. full size, original image
    # 2. half size, scaled down image
    # 3. thresholded image
    # 4. not rotated bounding box
    # 5. minimum area, rotated bounding box
    # 6. enclosing circle
    # 7. detected contour(s)

    # Read a sample image file, without changing it.  Adjust your path to match your location of the images folder.
    # Then, down-sample the input image using pyrDown()
    # Reference: https://docs.opencv.org/4.x/d4/d1f/tutorial_pyramids.html
    img = cv2.imread("../../../images/hammer.jpg", cv2.IMREAD_UNCHANGED)    # set path correctly for your installation.
    cv2.imshow("1. Source: Full Size", img)
    img = cv2.pyrDown(img)
    cv2.imshow("2. Source: Half Size", img)

    # Convert the color image to a grayscale image, then threshold.
    # Thresholding in THRESH_BINARY mode: if source pixel intensity is > thresh (127), then it is set to maxval (255)
    # Return values: ret is the threshold used (127) and thresh is the thresholded image.
    # Reference: https://docs.opencv.org/4.9.0/d7/d4d/tutorial_py_thresholding.html
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("3. Thresholded", thresh)

    # Find the contours in a binary image
    # Return values: contours is a list of the contours found,
    # hier[0] is a 2D array
    # hier[0][i] are the 4 hierarchical elements of contour i
    # hier[0][i][0] is the index of the next contour
    # hier[0][i][1] is the index of the previous contour
    # hier[0][i][2] is the index of the first child contour
    # hier[0][i][3] is the index of the parent contour
    # if any hier[0][i][j] is negative, the corresponding contour does not exist
    contours, hier = cv2.findContours(thresh,  # source image
                                      cv2.RETR_EXTERNAL,  # contour retrieval mode
                                      cv2.CHAIN_APPROX_SIMPLE)  # contour approximation method

    # Loop over all contours found
    for c in contours:
        # For each contour found:

        # ------------------------------------------------------------------

        # 1. find and draw bounding box (green)

        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("4. Green bounding box", img)
        areaBB = w * h
        print("Area of green bounding box:", areaBB)

        # ------------------------------------------------------------------

        # 2. find and draw the minimum area box (generally will be rotated) (red, with cyan corner dots)

        # minAreaRect() returns (X_center, Y_center), (Width, Height), Angle (degrees)
        rect = cv2.minAreaRect(c)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.intp(box)
        # draw contours
        cv2.drawContours(img,           # image
                         [box],         # an array of contours to draw
                         0,             # index of the contour to draw
                         (0, 0, 255),   # color to use
                         0)             # line thickness

        # mark each corner of the bounding box to understand the meaning of the values returned from boxPoints.
        radius = 3
        for i in box:
            cv2.circle(img, (i[0], i[1]), radius, (255, 255, 0), -1)
            radius *= 2  # double radius for each corner to be able to distinguish first, second, third, etc.

        cv2.imshow("5. Red minimum area bounding box", img)
        areaBBMin = int((rect[1][0] * rect[1][1]))
        print("Area of minimum area bounding box:", areaBBMin)
        print("Area ratio:", (areaBBMin / areaBB))

        # ------------------------------------------------------------------

        # 3. Find and draw the enclosing circle (green)

        # calculate the center and radius of minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # cast to integers
        center = (int(x), int(y))
        radius = int(radius)
        # draw the circle
        img = cv2.circle(img, center, radius, (0, 255, 0), 2)
        cv2.imshow("6. Green enclosing circle", img)

        # ------------------------------------------------------------------

    # 4. Draw all contours in blue
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    cv2.imshow("7. Contours", img)

    # ------------------------------------------------------------------

    # Make an image array of the same type as size as img.
    # Fill it with zeros.
    black = np.zeros_like(img)

    for c in contours:

        # arcLength() gives the perimeter length (distance around the object)
        # along the contour.
        # This is used to obtain a specification of the maximum
        # difference between the contour and the approximation, below
        epsilon = 0.01 * cv2.arcLength(c, True)
        print ("Maximum separation between contour and poly-line:", epsilon)

        # Approximate the contour with a curve having fewer points.
        # The approximation should be no farther than epsilon from the
        # original contour.
        approx = cv2.approxPolyDP(c, epsilon, True)

        # Calculate a convex hull around the contour.
        hull = cv2.convexHull(c)

        cv2.drawContours(black, [c], -1, (0, 255, 0), 2)        # draw the original contour in green
        cv2.drawContours(black, [approx], -1, (255, 255, 0), 2) # draw the approximate poly-line in cyan
        cv2.drawContours(black, [hull], -1, (0, 255, 255), 2)   # draw the convex hull in yellow

    cv2.imshow("8. Hull", black)

    # ------------------------------------------------------------------

    cv2.waitKey()
    cv2.destroyAllWindows()


def detectLines():
    # Detect the lines present in an image.

    # Read a sample image file, without changing it.  Adjust your path to match your location of the images folder.
    img = cv2.imread("../../../images/lines.jpg", cv2.IMREAD_UNCHANGED)    # set path correctly for your installation.
    cv2.imshow("1. Starting image", img)

    # Convert the color image to a grayscale image.
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("2. Grayscale image", grayImg)

    # Apply Canny filtering to obtain edges
    # If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge
    # If a pixel gradient value is below the lower threshold, then it is rejected.
    # If the pixel gradient is between the two thresholds, then it will be accepted only if
    # it is connected to a pixel that is above the upper threshold.

    t_upper = 600 # int(input("Enter upper Canny threshold: "))
    t_lower = 200 # int(input("Enter lower Canny threshold: "))
    edges = cv2.Canny(grayImg, t_lower, t_upper)
    cv2.imshow("3. Canny Filtered", edges)

    # Apply the probabilistic Hough transform to detect the lines.
    # Minimum line length. Line segments shorter than that are rejected.
    minLineLength = int(input("Enter minimum line length: ")) # 50

    # Maximum allowed gap between points on the same line to link them.
    maxLineGap = int(input("Enter maximum allowed gap in line: ")) # 20
    lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 20, minLineLength, maxLineGap)



    if lines is not None:
        sz = lines.size
        if sz > 0:
            (a,b,c) = lines.shape
            print (a,"lines detected.")
            for i in range(a):  # loop over lines
                (x1, y1, x2, y2) = lines[i][0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255,0), 2)    # draw green lines in the original image

            cv2.imshow("4. Lines", img)
    else:
        print("No lines detected.")

    # ------------------------------------------------------------------

    cv2.waitKey()
    cv2.destroyAllWindows()


def detectCircles():
    pass
