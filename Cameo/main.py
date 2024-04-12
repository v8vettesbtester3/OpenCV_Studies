import cv2
import filters
from managers import WindowManager, CaptureManager

class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

        self._LFilters = [{"title":"No filtering","filter":None}]

        # -------  To be added after creating filters in filters.py -------
        self._sharpenFilter = filters.SharpenFilter()
        self._LFilters.append({"title":"Sharpen Filter","filter":self._sharpenFilter})

        self._findEdgesFilter = filters.FindEdgesFilter()
        self._LFilters.append({"title":"Find-Edges Filter","filter":self._findEdgesFilter})

        self._blurFilter = filters.BlurFilter()
        self._LFilters.append({"title":"Blur Filter","filter":self._blurFilter})

        self._embossFilter = filters.EmbossFilter()
        self._LFilters.append({"title":"Emboss Filter","filter":self._embossFilter})
        # -----------------------------------------------------------------

    def runFilterDemo(self, choice):
        ''' Run the main loop. '''

        # Get a live image and filter it
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is not None:
                if choice == 0:
                    pass  # choice == 0: no filtering
                elif choice > 0 and choice < len(self._LFilters):
                    # Apply a chosen filter
                    filters.enhanceEdges(frame, frame)
                    self._LFilters[choice]["filter"].apply(frame, frame)
                elif choice == len(self._LFilters):
                    filters.enhanceEdges(frame, frame)
                elif choice == len(self._LFilters)+1:
                    frame = filters.cannyFilter(frame)
                    # for i in range(frame.shape[0]):
                    #     print(i,frame[i,0])
                    # input("Enter to continue")
                # add new filter calls here--------------------------
                else:
                    pass    # choice is out of range; no filtering

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        ''' Handle key press:
        space -> take a screenshot
        tab -> start/stop recording a screencast
        esc -> quit. '''
        if keycode == 32:   # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

def main():
    cam = Cameo()
    #for i in range(5):
    while True:
        print("Space\tscreenshot\nTab\t\tstart-stop screencast\nEscape\texit")

        # Present a menu
        print("\n"+"-"*15+"Action Menu"+"-"*15)    # print a menu
        for i in range(len(cam._LFilters)):
            print(i,cam._LFilters[i]["title"])

        idx = len(cam._LFilters)
        print(idx, "Enhance Edges")

        idx += 1
        print(idx, "Canny Filter")
        # add new menu items here ------------------------
        print()

        choice = -2
        while choice < -1 or choice > idx:  # get the user's choice
            msg = "Enter your choice (0-"+str(idx)+" -1=EXIT): "
            choice = int(input(msg))
        print()

        if (choice > -1):
            cam.runFilterDemo(choice)


            ans = input("Another? (y/n): ")
            if (ans != "y"): break

        else:
            break


if __name__ == '__main__':
    main()

