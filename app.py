import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# モデルの読み込み
model = load_model('mnist_model.h5')

st.title('手書き数字認識アプリ')
st.write('0から9までの手書き数字を認識するアプリです。')

# 記載をする場所
canvas = st.image(np.zeros((150, 150)), caption='手書き数字を描いてください.', use_column_width=True, channels='L')

# 「認識する」ボタンが押されたときの処理
if st.button('認識する'):
    # Canvas上の画像を読み込んでリサイズ
    img = canvas.image_data.astype(np.uint8)
    img = Image.fromarray(img).resize((28, 28)).convert('L')
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    # モデルで予測を実行
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # 予測結果を表示
    st.write(f'予測結果: {predicted_class}')