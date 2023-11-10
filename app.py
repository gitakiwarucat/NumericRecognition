import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image  # Imageクラスを追加
import cv2

# MNISTデータセットで訓練されたニューラルネットワークモデルをロード
model = tf.keras.models.load_model('mnist_model.h5')

# Streamlitアプリのタイトルを設定
st.title('数字の画像認識')

# 説明文
st.write("""
    このアプリでは、手書きで描いた数字を認識します。
    描画エリアに数字を描き、左下の保存ボタンを押下し、認識するボタンを押してください。
    （端末に画像が保存されることはありません。）
""")


# Canvasを表示
canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",  # Canvasの背景色を白に設定
    stroke_width=10,  # 描画時の線の太さを設定
    stroke_color="black",  # 描画時の線の色を設定
    background_color="#eee",  # Canvasの周りの背景色を設定
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# 手書き数字の認識処理を行う関数
def recognize_digit(image):
    # 画像をモデルに渡して予測を行う
    prediction = model.predict(image)
    # 予測結果からクラスを取得
    predicted_class = np.argmax(prediction)
    return predicted_class

# 「認識する」ボタンが押されたときの処理
if st.button('認識する'):
    # Canvas上の画像を読み込んでリサイズ
    img_array = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img_array)

    # 28x28にリサイズ
    resized_image = img.resize((28, 28))

    # リサイズした画像をNumPy配列に変換
    img_gray = cv2.cvtColor(np.array(resized_image), cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.reshape(1, 28, 28, 1) / 255.0  # 正規化と形状変更

    # 画像をモデルに渡して認識処理を行う
    predicted_class = recognize_digit(img_gray)

    # 認識結果を表示
    st.write(f'認識結果: {predicted_class}')
