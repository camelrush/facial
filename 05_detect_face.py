import face_recognition
import pickle
import cv2
import sys
import os

currentname = "unknown"
encodingsP = "encodings.pickle"
cascade = "haarcascade_frontalface_default.xml"

data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)


def detect_camera_face():

    # カメラを起動
    cap = cv2.VideoCapture(0)

    # 比較精度の初期値を設定
    tolerance = 0.4

    # カメラから1フレームずつ処理
    while cap.isOpened():

        # 1フレーム読み込み
        ret, frame = cap.read()

        # フレーム全体をグレースケール化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # カスケード分類器で顔を検出
        rects = detector.detectMultiScale(gray, scaleFactor=1.3,
                                          minNeighbors=3, minSize=(150, 150),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # 測定精度をフレームに表示
        cv2.putText(frame,
                    text=f'tolerance : {str(tolerance)}',
                    org=(1, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=2)

        # 同時に検出された顔画像分、繰り返す
        for box in boxes:

            encodings = face_recognition.face_encodings(rgb, [box])

            for encoding in encodings:

                # 顔画像と集積画像を照合。近似値から比較結果(T/F)を得る
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding, tolerance)

                # 集積画像の中にひとつでも類似画像があれば検出とする
                if True in matches:
                    
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)
                    color = (0, 0, 255)

                else:
                    name = "unknown"
                    color = (255, 255, 225)

                # 顔画像上に、検出された名前と四角枠を表示
                [top, right, bottom, left] = box
                cv2.rectangle(frame, (left, top), (right, bottom),
                              color, 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .8, color, 2)

                if name != "unknown":
                    break

        # 加工した1フレームを画面に表示
        cv2.imshow("realtime face detect window", frame)

        # キー入力を検知
        k = cv2.waitKey(1)

        # 'q'キー押下で終了
        if k & 0xFF == ord('q'):
            break
        # '+'キー押下でtoleranceを加算
        elif k & 0xFF == ord('+'):
            tolerance = round(tolerance + 0.1, 1)
        # '-'キー押下でtoleranceを減算
        elif k & 0xFF == ord('-'):
            tolerance = round(tolerance - 0.1, 1)

    # カメラリソース開放
    cap.release()
    cv2.destroyAllWindows()


def detect_imagefile_face(image_file):

    # 画像ファイルの存在チェック
    if os.path.exists(image_file) is False:
        print('invalid arguments.')
        print(f'image file {image_file} is not exists.')
        return

    # ファイルから画像を読み込み、RGB、グレースケールに変換
    frame = cv2.imread(image_file)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # RGB画像から顔を検出し、顔の座標を得る(CNNを使用)
    boxes = face_recognition.face_locations(gray, model="cnn")

    # 顔座標の矩形画像（つまり顔写真）を、数値リストに変換
    encodings = face_recognition.face_encodings(rgb, boxes)

    # 検出した顔それぞれを、集積画像と照合
    names = []
    for encoding in encodings:

        # 顔画像と集積画像を照合。近似値から比較結果(T/F)を得る
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding, 0.4)
        name = "Unknown"

        # 集積画像の中にひとつでも類似画像があれば検出とする
        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        # 名前を検出結果リストに追加
        names.append(name)

    # 検出画像分繰り返す
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # 顔画像上に、検出された名前と四角枠を表示
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 255, 255), 2)

    # 加工した画像を保存
    dname = \
        os.path.dirname(image_file) + '/' + \
        os.path.splitext(image_file)[0] + '_detected.' + \
        os.path.splitext(image_file)[1]
    cv2.imwrite(dname, frame)

    # 画像を画面に表示する
    print("show detected window...(see backend)")
    print("for exit, press any key on window.")
    cv2.imshow("detected image window", frame)
    cv2.waitKey(0)

    print("detect completed for " + os.path.basename(image_file))


if __name__ == "__main__":
    
    # カメラ検出モード
    if len(sys.argv) == 1:
        detect_camera_face()

    # イメージ画像検出モード
    elif len(sys.argv) == 2:
        detect_imagefile_face(sys.argv[1])

    # 引数エラー
    else:
        msg = ""
        msg += "invalid arguments. how to use.\n"
        msg += "  1. camera mode\n"
        msg += "     python 05_detect_face.py\n"
        msg += "  2. image file mode\n"
        msg += "     python 05_detect_face.py [jpg file name]\n"
        print(msg)
