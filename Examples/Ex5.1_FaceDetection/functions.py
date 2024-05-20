import cv2

def detectFaceStillImage():
    # Detects faces in a still image, using Haar cascades

    '''
    Haar features describe a pattern of contrast among adjacent image regions.
    The frontal face features describe contrast patterns typical in human faces.

    The features are organized into a hierarchy (cascade) with the highest layer
    corresponding to features having the greatest distinctiveness.
    '''
    face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

    # Input an example image containing frontal faces.
    img = cv2.imread("../../../images/woodcutters.jpg")

    # Convert from color to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create an image pyramid.
    # Rescale the image several times, with each successive scale increased by
    # a factor of 1.08.
    # Returns an array of rectangles which bound the detected faces.
    faces = face_cascade.detectMultiScale(gray, 1.08, 5)

# Loop over the rectangles around the detected faces.
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.namedWindow('Woodcutters Detected')

    cv2.imshow('Woodcutters Detected', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectFaceVideo():
    # Detects faces in a video image.  Draw blue rectangle around faces
    # green rectangles around eyes.

    '''
    Haar features describe a pattern of contrast among adjacent image regions.
    The frontal face features describe contrast patterns typical in human faces.

    The features are organized into a hierarchy (cascade) with the highest layer
    corresponding to features having the greatest distinctiveness.
    '''
    face_cascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("./cascades/haarcascade_eye.xml")

    #cv2.namedWindow('Video Faces')


    camera = cv2.VideoCapture(0)    # Select the camera for input

    while (cv2.waitKey(1) == -1):   # Loop until <esc> key is pressed
        success, frame = camera.read()  # grab an image

        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert color to grayscale

            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120,120))  # detect faces

            for (x,y,w,h) in faces:     # draw blue boxes around any faces detected
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

                roi_gray = gray[y:y+h, x:x+w]   # isolate further analysis to just a face
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, minSize = (40, 40))  # detect eyes therein

                for (ex, ey, ew, eh) in eyes:   # draw green boxes around any eyes detected
                    cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew, y+ey+eh), (0,255,0), 2)

            cv2.imshow('Video Faces', frame)    # show the annotated frame

    cv2.destroyAllWindows()     # after <esc> key pressed, close all windows.
