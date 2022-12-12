import cv2
import glob


def save_rotate(frame, photo, angle):
    (h, w) = frame.shape[:2]
    center = (w / 2, h / 2)
    scale = 1

    M = cv2.getRotationMatrix2D(center, angle, scale)
    frame_rotate = cv2.warpAffine(frame, M, (w, h))
    filename = photo.replace(".jpg", "_rotate_" + str(angle) + ".jpg")   
    cv2.imwrite(filename, frame_rotate)


# use this xml file
photos = glob.glob("./model_dataset/*/*.jpg")

for photo in photos:

    frame = cv2.imread(photo)

    print("rotate for ..." + photo)

    save_rotate(frame, photo, 30)
    save_rotate(frame, photo, 300)


photos = glob.glob("./model_dataset/*/*.jpg")

for photo in photos:

    frame = cv2.imread(photo)

    print("gray for ..." + photo)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filename = photo.replace(".jpg", "_gray" + ".jpg")
    cv2.imwrite(filename, frame_gray)


photos = glob.glob("./model_dataset/*/*.jpg")

for photo in photos:

    frame = cv2.imread(photo)

    print("mirror for ..." + photo)

    frame_mirror = cv2.flip(frame, 1)
    filename = photo.replace(".jpg", "_mirror" + ".jpg")
    cv2.imwrite(filename, frame_mirror)

