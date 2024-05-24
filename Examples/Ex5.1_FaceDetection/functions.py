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
    # Generate facial recognition training data

    # Captures several video frames from the camera, in rapid succession.
    # For each frame, an attempt is made to detect a face.
    # If a face is detected, a sub-image containing just that face is defined.
    # The sub-image is rescaled to be 200 X 200 pixels and written to a file.
    # Each thusly formed face image is written to a separate file.
    # Each such file is given a sequential name: 1.pgm, 2.pgm, 3.pgm, etc.
    # All files are stored at the path ../../../data/at/<user-supplied-initials>.pgm
    # .pgm files are Portable Gray Map files.  They are monochromatic (grayscale).

    # As a safeguard, this function will capture at most 100 images.  If the user
    # does not stop imaging before MAXCOUNT are obtained, the function will stop grabbing
    # images of its own accord.

    import os

    MAXCOUNT = 200  # maximum number of training images to capture

    initials = input("Enter your initials (no spaces): ")
    output_folder = '../../../data/at/'+initials      # path to folder containing generated images

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)      # create the destination folder if it does not exist

    # Set the Haar cascade that will be used to detect faces
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(0)    # connect to the camera

    count = 0

    # Grab images until either the user presses a key it the window or 100 images have been captured.
    while cv2.waitKey(1) == -1 and count < MAXCOUNT:
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


def read_images(path, image_size):
    # Load the training data for face recognition.

    # This function walks through a directory's subdirectories,
    # loads the images, resizes them to a specified size and puts
    # the resized images into a list.

    # Also, two other lists are built:
    # A list of names or initials and a list of labels (numeric IDs)
    # associated with the loaded images.
    # For example "abc" could be a name and 0 could be the label for
    # all of the images loaded from the abc subfolder.

    # The lists of labels and images are converted into numpy arrays.

    # The function returns three things:
    # 1. the list of names
    # 2. the numpy array of images
    # 3. the numpy array of labels


    import os
    import numpy

    names = []              # list of names
    training_images= []     # list of images
    training_labels = []    # list of labels

    label = 0               # initial label value

    for dirname, subdirnames, filenames in os.walk(path):
        for subdirname in subdirnames:
            names.append(subdirname)
            subject_path = str(os.path.join(dirname, subdirname))
            for filename in os.listdir(subject_path):
                img = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)

                if img is None:
                    # The file cannot be loaded as an image.
                    # Skip it.
                    continue

                img = cv2.resize(img, image_size)
                training_images.append(img)
                training_labels.append(label)

            label += 1

    training_images = numpy.asarray(training_images, numpy.uint8)
    training_labels = numpy.asarray(training_labels, numpy.int32)

    return names, training_images, training_labels


def recognizeFaces():
    path_to_training_images = '../../../data/at'
    #path_to_training_images = '..\\..\\..\\data\\at'
    training_image_size = (200, 200)
    names, training_images, training_labels = read_images(path_to_training_images, training_image_size)

    model = cv2.face.EigenFaceRecognizer.create()
    model.train(training_images, training_labels)

    # Set the Haar cascade that will be used to detect faces
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

    camera = cv2.VideoCapture(0)    # connect to the camera

    while cv2.waitKey(1) == -1:
        success, frame = camera.read()
        if success:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(120,120))

            for (x,y, w, h) in faces:  # Loop over the faces found in the frame
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)   # Draw a rectangle around the face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[y:y+h, x:x+w]   # isolate further analysis to just a face

                if gray is None:
                    # The file cannot be loaded as an image.
                    # Skip it.
                    continue

                roi_gray = cv2.resize(roi_gray, training_image_size)
                label, confidence = model.predict(roi_gray)

                text = '%s, confidence=%.2f' % (names[label], confidence)

                cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 2)

            cv2.imshow('Face Recognition', frame)

    cv2.destroyAllWindows()     # after <esc> key pressed, close all windows.

