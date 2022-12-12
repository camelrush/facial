import cv2
import glob
import dlib
import os


# use this xml file
cascade = "haarcascade_frontalface_default.xml"

detector = dlib.get_frontal_face_detector()

photos = glob.glob("./model_rawset/*/*.jpg")


def save(img, name, bbox, width=180, height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    # we need this line to reshape the images
    try:
        imgCrop = cv2.resize(imgCrop, (width, height))
        cv2.imwrite(name, imgCrop)
    except Exception as e:
        print(e)


for photo in photos:

    print("triming..." + photo)

    frame = cv2.imread(photo)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(gray)

    if len(faces) != 1:
        continue

    for counter, face in enumerate(faces):
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 255, 220), 1)
        # save(gray,new_path+str(counter),(x1-fit,y1-fit,x2+fit,y2+fit))
        save_path = photo.replace('model_rawset', 'model_dataset')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save(frame, save_path, (x1, y1, x2, y2))
