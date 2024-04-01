import cv2
import numpy as np
import os


def testFun():
    print("From testFun()")

def testImageDisplay():
    # display the penguin image
    image = cv2.imread("penguins.jpg")
    cv2.imshow('Penguins',image)
    print("Close image to exit function.")
    cv2.waitKey(0)

def imageAsAnArray():
    # define arrays and operate on them as images.
    # check the dimensions
    image = np.zeros((5,4), dtype=np.uint8) # 5 x 4 array of unsigned 8-bit integers, all = 0
    print("5 x 4 array of unsigned 8-bit integers, all = 0")
    print(image)
    print("The shape of this image: ", image.shape)

    print("\nNow convert this to a color image")
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)  # 5 x 4 array of 3 color planes, all = 0
    print("5 x 4 array of 3 color planes, all = 0")
    print(image)
    print("The shape of this image: ", image.shape)

def imageReading():
    '''
    A program to exhibit the behavior of several different image read modes.
    Loops through 12 different reading modes, displaying the result of
    reading a color jpeg image file for each mode.

    To run, press space bar after each image is displayed to close that image
    and display the next one.  As each image is displayed, the read mode is
    printed in the console.  Pressing space bar after the last image ends
    the program.

    Author: J. M. Hinckley
    Created: 22-Jan-2024
    '''

    # Default reading mode:
    image = cv2.imread('penguins.jpg')
    cv2.imshow('default', image)
    print("Default read mode.  Close image to continue.")
    cv2.waitKey(0)

    # Different image reading modes.
    modes = {
        # Default, 3-channel BGR, 8-bits per channel.
        'IMREAD_COLOR': cv2.IMREAD_COLOR,
        # Either 8-bit per channel BGR or 8-bit grayscale, depending on meta data in file.
        # Reads any possible color format.
        'IMREAD_ANYCOLOR': cv2.IMREAD_ANYCOLOR,
        # Produces 8-bit grayscale.
        'IMREAD_GRAYSCALE': cv2.IMREAD_GRAYSCALE,
        # Loads image as grayscale at its original depth.
        'IMREAD_ANYDEPTH': cv2.IMREAD_ANYDEPTH,
        # Loads image as BGR color at its original depth.
        'IMREAD_ANYDEPTH | IMREAD_COLOR': cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR,
        # Loads image as grayscale at half its original resolution.
        'IMREAD_REDUCED_GRAYSCALE_2': cv2.IMREAD_REDUCED_GRAYSCALE_2,
        # Loads image as 8-bit per channel BGR color, at half its original resolution.
        'IMREAD_REDUCED_COLOR_2': cv2.IMREAD_REDUCED_COLOR_2,
        # Loads image as grayscale at one-quarter its original resolution.
        'IMREAD_REDUCED_GRAYSCALE_4': cv2.IMREAD_REDUCED_GRAYSCALE_4,
        # Loads image as 8-bit per channel BGR color, at one-quarter its original resolution.
        'IMREAD_REDUCED_COLOR_4': cv2.IMREAD_REDUCED_COLOR_4,
        # Loads image as grayscale at one-eighth its original resolution.
        'IMREAD_REDUCED_GRAYSCALE_8': cv2.IMREAD_REDUCED_GRAYSCALE_8,
        # Loads image as 8-bit per channel BGR color, at one-eighth its original resolution.
        'IMREAD_REDUCED_COLOR_8': cv2.IMREAD_REDUCED_COLOR_8
    }

    # Loop over reading modes and display the image in each mode.
    for modeKey in modes.keys():
        image = cv2.imread('penguins.jpg', modes[modeKey])
        cv2.imshow(modeKey, image)
        print(modeKey,'read mode.  Close image to continue.')
        cv2.waitKey(0)


def rawBytes():
    '''
    A function to demonstrate the creation of a bytearray and then the conversion
    of that into a numpy array, which is the form of an OpenCV image.  This uses
    a grayscale image

    The elements of the bytearray are selectively set to zero to create
    specific patterns in the resulting image.  This demonstrates the relationship
    between the bytearray indices and the image pixel locations.

    To run, enter a number between 1 and 6 to create one of the pixel patterns.
    Generally, the array is looped over, calculating the corresponding row
    and column index values.  Patterns in the image are created by selectively
    setting array values to zero, based on the row and column index values.

    After the image is shown, put the caret (cursor) in the image and press
    the Escape key to advance to the next presentation of the menu, to
    create a new pattern.  End the program by entering 0 (zero) in the menu.

    Author: J. M. Hinckley
    Created: 25-Jan-2024
    '''

    # Image dimensions
    height = 300
    width = 400

    choice = -1
    while choice != 0:

        # Print menu
        choice = -1
        while choice < 0 or choice > 6:
            print('Enter choice of operation:')
            print('1: zero pixels in lower half of image')
            print('2: zero pixels in right half of image')
            print('3: zero pixels in upper right corner of image')
            print('4: zero pixels in central circular region of image')
            print('5: zero pixels on border of image')
            print('6: zero pixels in central rectangular region of image')
            print('0: quit')
            schoice = input('')
            choice = int(schoice)

        # make an array of <height*width> random bytes
        randomByteArray = bytearray(os.urandom(height*width))

        if choice == 1:
            # replace pixel values in bottom half of image
            for i in range(height*width//2,height*width):
                randomByteArray[i] = 0

        elif choice == 2:
            # replace pixel values in right half of image
            # Get row, column indices
            for i in range(height*width):
                row = i // width
                col = i % width
                if col >= width//2:
                    randomByteArray[i] = 0

        elif choice == 3:
            # replace pixel values in upper right corner of image
            # Get row, column indices
            for i in range(height*width):
                row = i // width
                col = i % width
                if row < col:
                    randomByteArray[i] = 0

        elif choice == 4:
            # replace pixel values in circular center region of image
            # Get row, column indices
            for i in range(height*width):
                row = i // width
                col = i % width
                ccenter = width // 2
                rcenter = height // 2
                radius = min(height, width) // 4
                if (row-rcenter)**2 + (col-ccenter)**2 < radius**2:
                    randomByteArray[i] = 0

        elif choice == 5:
            # replace pixel values around edge of image
            # Get row, column indices
            for i in range(height*width):
                row = i // width
                col = i % width
                border = 50
                if row < border or row > (height-border) \
                        or col < border or col > (width-border):
                    randomByteArray[i] = 0

        elif choice == 6:
            # replace pixel values in central rectangular area of image
            # Get row, column indices
            for i in range(height*width):
                row = i // width
                col = i % width
                border = 50
                if row >= border and row <= (height-border) \
                        and col >= border and col <= (width-border):
                    randomByteArray[i] = 0

        else:
            continue    # invalid choice, just loop.


        flatNumpyArray = np.array(randomByteArray)

        # convert the array to make a <height> x <width> grayscale image
        grayImage = flatNumpyArray.reshape(height,width)
        cv2.imshow('grayscale', grayImage)
        cv2.waitKey(0)


def rawBytesColor():
    '''
    A function to demonstrate the creation of a bytearray and then the conversion
    of that into a numpy array, which is the form of an OpenCV image.  This uses
    a color image.  For each pixel, 3 bytes are used to represent the values in
    each of the 3 color channels.  The first byte is the blue value, the second
    byte is the green value and the third byte is the red value.

    The elements of the bytearray are selectively set to zero to create
    specific patterns in the resulting image.  This demonstrates the relationship
    between the bytearray indices and the image pixel locations.

    To run, enter a number between 1 and 6 to create one of the pixel patterns.
    Generally, the array is looped over, calculating the corresponding row
    and column index values.  Patterns in the image are created by selectively
    setting array values to zero, based on the row and column index values.

    After the image is shown, put the caret (cursor) in the image and press
    the Escape key to advance to the next presentation of the menu, to
    create a new pattern.  End the program by entering 0 (zero) in the menu.

    Author: J. M. Hinckley
    Created: 25-Jan-2024
    '''

    # different operations available
    ops = ['1: zero pixels in lower half of image',
           '2: zero pixels in right half of image',
           '3: zero pixels in upper right corner of image',
           '4: zero pixels in central circular region of image',
           '5: zero pixels on border of image',
           '6: zero pixels in central rectangular region of image',
           '7: zero red channel in pixel values in bottom half of image',
           '8: zero green channel in pixel values in bottom half of image',
           '9: zero blue channel in pixel values in bottom half of image',
           ]
    numOps = len(ops) # number of different operations available in menu

    # Image dimensions
    height = 300
    width = 400

    choice = -1
    while choice != 0:  # 0 exits loop

        # Print menu
        choice = -1
        while choice < 0 or choice > numOps:  # while invalid choice
            print('Enter choice of operation:')
            for i in range(numOps):
                print(ops[i])
            print('0: quit')
            schoice = input('')
            choice = int(schoice)

        # make an array of <height*width> random bytes
        randomByteArray = bytearray(os.urandom(height*width*3))

        if choice == 1:
            # replace pixel values in bottom half of image
            # Get row, column indices
            for i in range(height*width*3):
                row = i // 3 // width
                col = (i // 3) % width
                if row >= height//2:
                    randomByteArray[i] = 0

        elif choice == 2:
            # replace pixel values in right half of image
            # Get row, column indices
            for i in range(height*width*3):
                row = i // 3 // width
                col = (i // 3) % width
                if col >= width//2:
                    randomByteArray[i] = 0

        elif choice == 3:
            # replace pixel values in upper right corner of image
            # Get row, column indices
            for i in range(height*width*3):
                row = i // 3 // width
                col = (i // 3) % width
                if row < col:
                    randomByteArray[i] = 0

        elif choice == 4:
            # replace pixel values in circular center region of image
            # Get row, column indices
            for i in range(height*width*3):
                row = i // 3 // width
                col = (i // 3) % width
                ccenter = width // 2
                rcenter = height // 2
                radius = min(height, width) // 4
                if (row-rcenter)**2 + (col-ccenter)**2 < radius**2:
                    randomByteArray[i] = 0

        elif choice == 5:
            # replace pixel values around edge of image
            # Get row, column indices
            for i in range(height*width*3):
                row = i // 3 // width
                col = (i // 3) % width
                border = 50
                if row < border or row > (height-border) \
                        or col < border or col > (width-border):
                    randomByteArray[i] = 0

        elif choice == 6:
            # replace pixel values in central rectangular area of image
            # Get row, column indices
            for i in range(height*width*3):
                row = i // 3 // width
                col = (i // 3) % width
                border = 50
                if row >= border and row <= (height-border) \
                        and col >= border and col <= (width-border):
                    randomByteArray[i] = 0


        elif choice == 7 or choice == 8 or choice == 9:
            # zero red channel in pixel values in bottom half of image
            # Get row, column indices
            for i in range(height * width * 3):
                row = i // 3 // width
                col = (i // 3) % width
                if row >= height // 2:
                    if choice == 7:
                        if i % 3 == 2:  # zero red channel
                            randomByteArray[i] = 0
                    elif choice == 8:
                        if i % 3 == 1:  # zero green channel
                            randomByteArray[i] = 0
                    else:
                        if i % 3 == 0:  # zero blue channel
                            randomByteArray[i] = 0
        else:
            continue    # invalid choice, just loop.


        flatNumpyArray = np.array(randomByteArray)

        # convert the array to make a <height> x <width> grayscale image
        image = flatNumpyArray.reshape(height,width,3)
        cv2.imshow('color', image)
        cv2.waitKey(0)
