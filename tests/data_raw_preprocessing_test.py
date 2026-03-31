import os
import numpy as np
from PIL import Image
import cv2

# Ścieżki do katalogów
input_dir = 'data/raw'  # Katalog, gdzie znajdują się oryginalne dane
output_dir = 'data/augmented'  # Katalog, do którego zapiszemy przetworzone obrazy

# Funkcja do augmentacji obrazu
def augment_image(image):
    augmented_images = []
    
    # Przesunięcie
    shift_x, shift_y = 10, 5
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(np.array(image), matrix, (image.width, image.height))
    augmented_images.append(Image.fromarray(shifted_image))
    
    # Obrót
    rotated_image = image.rotate(15)  # rotacja o 15 stopni
    augmented_images.append(rotated_image)
    
    # Skalowanie
    scaled_image = image.resize((int(image.width * 0.8), int(image.height * 0.8)))
    augmented_images.append(scaled_image)
    
    # Dodanie szumu
    noise = np.random.normal(0, 25, (image.height, image.width, 3))
    noisy_image = np.array(image) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    augmented_images.append(Image.fromarray(noisy_image))
    
    return augmented_images

# Przetwarzanie obrazów we wszystkich folderach
for folder_name in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder_name)
    if os.path.isdir(folder_path):
        output_folder = os.path.join(output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(file_path)
                
                # Zapis oryginalnego obrazu
                image.save(os.path.join(output_folder, file_name))
                
                # Tworzenie i zapis augmentowanych wersji
                augmented_images = augment_image(image)
                for i, aug_image in enumerate(augmented_images):
                    aug_image.save(os.path.join(output_folder, f"{file_name.split('.')[0]}_aug{i+1}.jpg"))

print("Augmentacja zakończona pomyślnie.")
