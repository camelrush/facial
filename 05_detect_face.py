import face_recognition
import pickle
import cv2
import sys
import os

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
# use this xml file
cascade = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)


def detect_camera_face():

    cap = cv2.VideoCapture(0)

    tolerance = 0.4

    while cap.isOpened():
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.3,
                                          minNeighbors=3, minSize=(150, 150),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # show tolerance
        cv2.putText(frame,
                    text=f'tolerance : {str(tolerance)}',
                    org=(1, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2)

        for box in boxes:

            encodings = face_recognition.face_encodings(rgb, [box])

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding, tolerance)
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)
                    color = (0, 0, 255)

                else:
                    name = "unknown"
                    color = (255, 255, 225)

                # draw the predicted face name on the image
                #  - color is in BGR
                [top, right, bottom, left] = box
                cv2.rectangle(frame, (left, top), (right, bottom),
                              color, 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .8, color, 2)

        cv2.imshow("realtime face detect window", frame)

        k = cv2.waitKey(1)
        # press 'q'. quit
        if k & 0xFF == ord('q'):
            break
        # press '+'. tolerance up.
        elif k & 0xFF == ord('+'):
            tolerance = round(tolerance + 0.1, 1)
        # press '-'. tolerance down.
        elif k & 0xFF == ord('-'):
            tolerance = round(tolerance - 0.1, 1)

    cap.release()
    cv2.destroyAllWindows()


def detect_imagefile_face(image_file):

    # check image file exist.
    if os.path.exists(image_file) is False:
        print('invalid arguments.')
        print(f'image file {image_file} is not exists.')
        return

    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    frame = cv2.imread(image_file)

    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    boxes = face_recognition.face_locations(gray, model="cnn")

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding, 0.4)
        name = "Unknown"  # if face is not recognized, then print Unknown

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 255, 255), 2)

    # save image ([image_file]_detected.jpg)
    dname = \
        os.path.dirname(image_file) + '/' + \
        os.path.splitext(image_file)[0] + '_detected.' + \
        os.path.splitext(image_file)[1]
    cv2.imwrite(dname, frame)

    # display the image to our screen
    print("show detected window...(see backend)")
    print("for exit, press any key on window.")
    cv2.imshow("detected image window", frame)
    cv2.waitKey(0)

    print("detect completed for " + os.path.basename(image_file))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # mode : camera detection
        detect_camera_face()
    elif len(sys.argv) == 2:
        # mode : image file detection
        detect_imagefile_face(sys.argv[1])
    else:
        msg = ""
        msg += "invalid arguments. how to use.\n"
        msg += "  1. camera mode\n"
        msg += "     python 05_detect_face.py\n"
        msg += "  2. image file mode\n"
        msg += "     python 05_detect_face.py [jpg file name]\n"
        print(msg)
