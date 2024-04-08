from functions import *

def main():
    # Create a menu to exercise the many functions in the functions module

    LFuncs = [{"title":"EXIT","func":None},
              {"title":"High Pass Filtering",    "func":highPassFiltering},
              #{"title":"Test image display",            "func":testImageDisplay},
              #{"title":"Image as an array",             "func":imageAsAnArray},
              #{"title":"Exhibit image reading modes",   "func":imageReading},
              #{"title":"Grayscale image from random bytes", "func": rawBytes},
              #{"title":"Color image from random bytes", "func": rawBytesColor},
              #{"title":"Image as an array: subregions", "func":imageAsAnArray2},
              #{"title":"Region of interest", "func": regionOfInterest},
              ]


    while True:                      # continue until an exit is requested
        print("\n"+"-"*15+"Action Menu"+"-"*15)    # print a menu
        for i in range(len(LFuncs)):
            print(i,LFuncs[i]["title"])
        print()

        choice = -1
        while choice < 0 or choice >= len(LFuncs):  # get the user's choice
            msg = "Enter your choice (0-"+str(len(LFuncs)-1)+"): "
            choice = int(input(msg))
        print()

        if choice != 0:
            LFuncs[choice]["func"]()    # run the selected function
        else:
            break                       # Exit requested, so leave

    print("Done.")

if __name__ == '__main__':
    main()
