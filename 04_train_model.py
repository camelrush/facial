from imutils import paths
import face_recognition
import pickle
import cv2
import os

print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("model_dataset"))

knownEncodings = []
knownNames = []

# フォルダ内の画像ファイル分、繰り返す
for (i, imagePath) in enumerate(imagePaths):

    # フォルダ名をイメージラベル'name'とする
    print("[INFO] processing image {}/{} {}".format(i +
          1, len(imagePaths), imagePath), " ")
    name = imagePath.split(os.path.sep)[-2]

    # ファイルから画像を読み込み、RGB色に変換
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # RGB画像から顔を検出し、顔の座標を得る(CNNを使用)
    boxes = face_recognition.face_locations(rgb, model="cnn")
    print(len(boxes))

    # 顔座標の矩形画像（つまり顔写真）を、数値リストに変換
    encodings = face_recognition.face_encodings(rgb, boxes)

    # 得られた数値リストとイメージラベルを、それぞれリストに保存
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# binary'encodings.pickle'ファイルに、モデルデータを保存
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
