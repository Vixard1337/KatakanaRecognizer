import tensorflow as tf

# Ładowanie modelu
model = tf.keras.models.load_model('katakana_cnn_model.h5')

# Ścieżka do danych testowych
data_dir = '/home/admin/Dyplom/katakana/data/processed/augmented'
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Generator danych testowych
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(100, 100),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

# Ewaluacja modelu
loss, accuracy = model.evaluate(test_generator)
print(f"Loss: {loss}, Accuracy: {accuracy}")
