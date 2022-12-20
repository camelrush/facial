import cv2
import glob


# 1.-30°、30°に傾けた画像を作成
photos = glob.glob("./model_dataset/*/*.jpg")
for photo in photos:

    frame = cv2.imread(photo)

    print("rotate for ..." + photo)

    for angle in [30, -30]:
        (h, w) = frame.shape[:2]
        center = (w / 2, h / 2)
        scale = 1

        M = cv2.getRotationMatrix2D(center, angle, scale)
        frame_rotate = cv2.warpAffine(frame, M, (w, h))
        filename = photo.replace(".jpg", "_rotate_" + str(angle) + ".jpg")   
        cv2.imwrite(filename, frame_rotate)


# 2.グレースケールに変色させた画像を作成
photos = glob.glob("./model_dataset/*/*.jpg")
for photo in photos:

    frame = cv2.imread(photo)

    print("gray for ..." + photo)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filename = photo.replace(".jpg", "_gray" + ".jpg")
    cv2.imwrite(filename, frame_gray)


# 3.水平方向に反転させた画像を作成
photos = glob.glob("./model_dataset/*/*.jpg")
for photo in photos:

    frame = cv2.imread(photo)

    print("mirror for ..." + photo)

    frame_mirror = cv2.flip(frame, 1)
    filename = photo.replace(".jpg", "_mirror" + ".jpg")
    cv2.imwrite(filename, frame_mirror)

