# face_recognitionによる顔画像認識

## Python　AI ライブラリの分類(カテゴリ)

python 


## 操作方法

### モデルデータ作成

- モデル作成用のjpg画像を、ディレクトリ「model_dataset/(モデル名)」に格納する。`例：model_dataset/jun/face001.jpg、…/face002.jpg、…`
- 以下のコマンドでモデルデータを作成する。正常に終了した場合は、モデルデータ`encodings.pickle`ファイルが作成される。
  > python train_model.py

### 検出テスト
- 検出用のjpg画像を、ディレクトリ「photo」に格納する。複数枚可能。
- 以下のコマンドで、画像からモデルデータ作成した対象者の顔検出を実行する。
  > python detect_face.py
- 検出した結果は、ディレクトリ「photo_detected」に格納される。検出用のjpg画像の中で、顔と識別された箇所には黄色い□枠が表示され、モデルデータと一致した場合波枠の上部にモデル名が表示される。

