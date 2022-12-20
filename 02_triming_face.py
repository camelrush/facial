import cv2
import glob
import dlib
import os

detector = dlib.get_frontal_face_detector()
photos = glob.glob("./model_rawset/*/*.jpg")

# 最大縦横サイズ
MAX_TRIM_SIZE = 500


def save(img, name, bbox):

    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]

    try:
        cv2.imwrite(name, imgCrop)
    except Exception as e:
        print(e)


for photo in photos:

    print("triming for " + photo + ".")

    # ファイルから画像を読み込む
    frame = cv2.imread(photo)

    # サイズが大きすぎる場合は、最大サイズまで縮小する
    h, w, _ = frame.shape
    if max([h, w]) > MAX_TRIM_SIZE:
        ratio = MAX_TRIM_SIZE / max([h, w])
        frame = cv2.resize(frame, dsize=None, fx=ratio, fy=ratio)

    # 検出用画像を取得(グレー)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 出力用画像を取得(カラー)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 顔部分を検出(dlib)
    faces = detector(gray)

    # 1人の画像以外は除く
    if len(faces) != 1:
        continue

    # 検出された顔の範囲を画像として保存する
    face = faces[0]
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()

    save_path = photo.replace('model_rawset', 'model_dataset')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save(frame, save_path, (x1, y1, x2, y2))
