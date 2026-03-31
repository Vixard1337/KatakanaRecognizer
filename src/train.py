import tensorflow as tf
from model import create_model

# Parametry sieci
input_shape = (100, 100, 1)
num_classes = 46
filters = 64
kernel_size = 3
num_layers = 3
epochs = 30

# Ścieżka do danych
data_dir = '/home/admin/Dyplom/katakana/data/processed/augmented'
weights_path = 'weights.h5'

# Przygotowanie generatora danych
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(100, 100),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(100, 100),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Tworzenie i kompilacja modelu
model = create_model(input_shape, num_classes, filters, kernel_size, num_layers)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacki
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_katakana_model.h5', save_best_only=True, monitor='val_loss', mode='min')

try:
    model.load_weights(weights_path)
    print("Wagi zostały załadowane.")
except:
    print("Nie udało się załadować wag, trenowanie od początku.")

# Trenowanie modelu
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint]
)
model.save_weights(weights_path)
print("Wagi zostały zapisane.")

# Zapisanie modelu
model.save('katakana_cnn_model.h5')
