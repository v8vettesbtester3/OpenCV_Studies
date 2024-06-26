"""
Program: ExpProcImages

A program to demonstrate image processing operations.

Author: J. M. Hinckley
Created: 2024
"""

from functions import *


def main():
    # Create a menu to exercise the many functions in the functions module

    LFuncs = [{"title": "EXIT", "func": None},
              {"title": "High Pass Filtering", "func": highPassFiltering},
              {"title": "Canny Filtering", "func": cannyFiltering},
              {"title": "Find a contour on a square", "func": contour1},
              {"title": "Find a contour on an image", "func": contour2},
              {"title": "Detect lines in an image", "func": detectLines},
              {"title": "Detect circles in an image", "func": detectCircles},
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
