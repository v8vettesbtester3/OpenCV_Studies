"""
Program: FaceDetection

A program to demonstrate face detection operations.

Author: J. M. Hinckley
Created: 2024
"""

from functions import *


def main():
    # Create a menu to exercise the many functions in the functions module

    LFuncs = [{"title": "EXIT", "func": None},
              {"title": "Detect faces in still image", "func": detectFaceStillImage},
              {"title": "Detect face in video", "func": detectFaceVideo},
              {"title": "Generate data for face recognition", "func": genFaceData},
              {"title": "Face recognition", "func": recognizeFaces},
              ]

    while True:  # continue until an exit is requested
        print("\n" + "-" * 15 + "Action Menu" + "-" * 15)  # print a menu
        for i in range(len(LFuncs)):
            print(i, LFuncs[i]["title"])
        print()

        choice = -1
        while choice < 0 or choice >= len(LFuncs):  # get the user's choice
            msg = "Enter your choice (0-" + str(len(LFuncs) - 1) + "): "
            choice = int(input(msg))
        print()

        if choice != 0:
            LFuncs[choice]["func"]()  # run the selected function
        else:
            break  # Exit requested, so leave

    print("Done.")


if __name__ == '__main__':
    main()
