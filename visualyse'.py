from flask import Flask, request, send_from_directory
import cv2
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    filename = 'uploaded_image.png'
    file.save(filename)
    MIN_MATCH_COUNT = 10
    # Загрузка исходного изображения и изображения для сравнения
    img1 = Image.open('DALL3.png')  # Исходное изображение
    img2 = Image.open(filename)

    img1 = np.array(img1.convert('L'))
    img2 = np.array(img2.convert('L'))

    # Инициализация SIFT детектора
    sift = cv2.SIFT_create()

    # Нахождение ключевых точек и дескрипторов для обоих изображений
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Инициализация BFMatcher
    bf = cv2.BFMatcher()

    # Сопоставление дескрипторов с использованием BFMatcher и KNN
    matches = bf.knnMatch(des1, des2, k=2)

    # Отсеивание ложных совпадений с использованием теста отношения Лоу
    good = []
    for m, n in matches:
        if m.distance < 0.35 * n.distance:
            good.append([m])

    # Проверка наличия совпадений
    if len(good) > MIN_MATCH_COUNT:
        return 'Успешно', 200
    else:
        return 'Неудачно', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
