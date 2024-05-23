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

    cv2.namedWindow('Video Faces')


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


def genFaceData():
    # Captures several video frames from the camera, in rapid succession.
    # For each frame, an attempt is made to detect a face.
    # If a face is detected, a sub-image containing just that face is defined.
    # The sub-image is rescaled to be 200 X 200 pixels and written to a file.
    # Each thusly formed face image is written to a separate file.
    # Each such file is given a sequential name: 1.pgm, 2.pgm, 3.pgm, etc.
    # All files are stored at the path ../../../data/at/<user-supplied-initials>.pgm
    # .pgm files are Portable Gray Map files.  They are monochromatic (grayscale).

    # As a safeguard, this function will capture at most 100 images.  If the user
    # does not stop imaging before 100 are obtained, the function will stop grabbing
    # images of its own accord.

    import os

    initials = input("Enter your initials (no spaces): ")
    output_folder = '../../../data/at/'+initials      # path to folder containing generated images

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)      # create the destination folder if it does not exist

    # Set the Haar cascade that will be used to detect faces
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(0)    # connect to the camera

    count = 0

    # Grab images until either the user presses a key it the window or 100 images have been captured.
    while cv2.waitKey(1) == -1 and count < 100:
        success, frame = camera.read()
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120,120))

            for (x,y, w, h) in faces:  # Loop over the faces found in the frame
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)   # Draw a rectangle around the face
                face_img = cv2.resize(gray[y:y+h, x:x+w], (200,200))    # Resize the sub-image of the face
                face_filename = '%s/%d.pgm' % (output_folder, count)    # Consecutive file names
                cv2.imwrite(face_filename, face_img)    # Write the image to the named file.
                count += 1      # Number for the next image file

            cv2.imshow('Capturing faces...', frame)

    cv2.destroyAllWindows()     # after <esc> key pressed, close all windows.
