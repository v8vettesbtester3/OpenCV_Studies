import cv2
from pprint import pprint

def is_fourcc_available(codec):
    try:
        fourcc = cv2.VideoWriter.fourcc(*codec)
        temp_video = cv2.VideoWriter('temp.mkv', fourcc, 30, (640, 480), isColor=True)
        return temp_video.isOpened()
    except:
        return False

def enumerate_fourcc_codecs():
    codecs_to_test = ["DIVX", "XVID", "MJPG", "WMV1", "WMV2", "FMP4","mp4v", "avc1", "I420", "IYUV", "mpg1", "H264"]
    available_codecs = []
    for codec in codecs_to_test:
        try:
            available_codecs.append((codec, is_fourcc_available(codec)))
        except Exception:
            print("skipping",codec)
            available_codecs.append((codec, False))

    return available_codecs

def copyVideo_Method1():

    codecs = enumerate_fourcc_codecs()
    print("Available FourCC codecs:")
    pprint(codecs)

    videoCapture = cv2.VideoCapture('file_example_AVI_640_800kB.avi')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = [

    # Create a raw video
    cv2.VideoWriter('1_rawVideo.avi',     # output video file name;
                                                        # any existing file with same name is overwritten
                                  0,                    # video codec
                                  fps,                  # frame rate
                                  size),                 # frame size

    # Uncompressed YUV encoding, 4:2:0 subsampling
    cv2.VideoWriter('2_YUV420_I420.avi',  # output video file name;
                                                        # any existing file with same name is overwritten
                                  cv2.VideoWriter.fourcc('I','4','2','0'),  # video codec
                                  fps,                  # frame rate
                                  size),                 # frame size

    # MPEG - 1
    cv2.VideoWriter('3_mpeg1_PIM1.avi',   # output video file name;
                                  # any existing file with same name is overwritten
                                  cv2.VideoWriter.fourcc('P', 'I', 'M', '1'),  # video codec
                                  fps,                  # frame rate
                                  size),                 # frame size

    # Old MPEG - 4
    cv2.VideoWriter('4_mpeg4Old_XVID.avi',# output video file name;
                                  # any existing file with same name is overwritten
                                  cv2.VideoWriter.fourcc('X', 'V', 'I', 'D'),  # video codec
                                  fps,                  # frame rate
                                  size),                 # frame size


    # Old MPEG - 4
    cv2.VideoWriter('5_mpeg4Old_MP4V.mp4',# output video file name;
                                  # any existing file with same name is overwritten
                                  cv2.VideoWriter.fourcc(*'mp4v'),  # video codec
                                  fps,                  # frame rate
                                  size),                 # frame size

    # New MPEG - 4
    #cv2.VideoWriter('6_mpeg4New_X264.mp4',# output video file name;
    cv2.VideoWriter('6_mpeg4New_H264.mp4',  # output video file name;
                                    # any existing file with same name is overwritten
                                  cv2.VideoWriter.fourcc(*'avc1'),  # video codec
                                  fps,                  # frame rate
                                  size),                 # frame size
        ]


    flags = [True for i in range(len(videoWriter))]  # used to control how many times message is printed

    success, frame = videoCapture.read()  # read a frame from the input video file

    print("Starting video writing.")
    while success:
        for i in range(len(videoWriter)):
            if videoWriter[i].isOpened():
                videoWriter[i].write(frame)
            else:
                if flags[i]:  # print the message just once
                    print("Codec not supported: ", i)
                    flags[i] = False
        success, frame = videoCapture.read()  # read a frame from the input video file

    print("Finished video writing.")



def capture():
    cameraCapture = cv2.VideoCapture(0)
    fps = 30
    size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('MyOutputVid.mp4', cv2.VideoWriter.fourcc(*'mp4v'),
                                  fps, size)

    duration = int(input("Enter capture duration (sec): ")) # seconds
    maxFrames = duration * fps
    numTicks = maxFrames // duration
    print("Video capture started.")
    print("="*numTicks)
    success, frame = cameraCapture.read()
    numFramesRemaining = maxFrames - 1
    while success and numFramesRemaining > 0:
        videoWriter.write(frame)
        success, frame = cameraCapture.read()
        if (numFramesRemaining % duration == 0):
            print('*',end="")
        numFramesRemaining -= 1
    print("*")
    print("Video capture ended.")

def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


def displayLiveImage():
    global clicked
    clicked = False

    cameraCapture = cv2.VideoCapture(0)

    cv2.namedWindow('The Window')
    cv2.setMouseCallback('The Window', onMouse)

    print("Showing camera feed.  Click window or press any key to stop.")
    success, frame = cameraCapture.read()
    while success and cv2.waitKey(1) == -1 and not clicked:
        cv2.imshow('The Window', frame)
        success, frame = cameraCapture.read()

    cv2.destroyWindow('The Window')
    cameraCapture.release()
