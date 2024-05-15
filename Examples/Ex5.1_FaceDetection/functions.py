import cv2

def detectFaceStillImage():
    # Detects faces in a still image, usnig Haar cascades

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

