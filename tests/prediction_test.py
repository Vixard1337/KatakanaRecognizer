import os
import numpy as np
from PIL import Image
from src.model import create_katakana_cnn
import cv2

# Ścieżka do obrazu z testowym znakiem "ア"
IMAGE_PATH = 'katakana_images/ア.jpg'

# Wczytanie modelu (podaj ścieżkę do swojego modelu)
MODEL_PATH = 'katakana/src/model.py'
model = create_katakana_cnn(MODEL_PATH)

# Mapowanie etykiet klas do znaków Katakany
katakana_labels = [
    'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ',
    'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト',
    'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ',
    'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ',
    'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン'
]

# Wczytanie obrazu i przetworzenie go na odpowiedni format
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Konwersja do skali szarości
    image = image.resize((64, 64))  # Dopasowanie rozmiaru obrazu do wymaganego przez model
    image_array = np.array(image) / 255.0  # Normalizacja
    image_array = np.expand_dims(image_array, axis=-1)  # Dodanie kanału
    image_array = np.expand_dims(image_array, axis=0)  # Dodanie wymiaru batch
    return image_array

# Przetworzenie obrazu testowego
test_image = preprocess_image(IMAGE_PATH)

# Przewidywanie klasy
predictions = model.predict(test_image)
predicted_index = np.argmax(predictions)
predicted_label = katakana_labels[predicted_index]

# Sprawdzanie wyniku
if predicted_label == 'ア':
    print("Sukces: Program rozpoznał znak 'ア' jako 'ア'")
else:
    print(f"Niepowodzenie: Program rozpoznał znak 'ア' jako '{predicted_label}'")
