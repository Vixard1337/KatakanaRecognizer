import tensorflow as tf
from tensorflow.python.client import device_lib

def check_cuda():
    # Check if GPU devices are available
    print("Czy TensorFlow wykrywa GPU:")
    print(tf.config.list_physical_devices('GPU'))

def check_keras():
    # Simple Keras model to check Keras/TensorFlow availability
    print("\nTestowanie Keras/TensorFlow...")
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(3,), activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.summary()
        print("Keras/TensorFlow działają poprawnie!")
    except Exception as e:
        print("Błąd podczas testu Keras/TensorFlow:", e)

def check_cuda_version():
    # Check CUDA version
    try:
        print("\nWersja CUDA:")
        cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
        print(cuda_version)
    except KeyError:
        print("CUDA nie jest dostępna lub TensorFlow nie jest skompilowane z obsługą CUDA.")

def check_cudnn_version():
    # Check cuDNN version
    try:
        print("\nWersja cuDNN:")
        cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
        print(cudnn_version)
    except KeyError:
        print("cuDNN nie jest dostępne lub TensorFlow nie jest skompilowane z obsługą cuDNN.")

if __name__ == "__main__":
    check_cuda()
    check_keras()
    check_cuda_version()
    check_cudnn_version()