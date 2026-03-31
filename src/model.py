import tensorflow as tf

def create_model(input_shape, num_classes, filters=64, kernel_size=3, num_layers=3, dropout_rate=0.5, learning_rate=0.001):
    model = tf.keras.models.Sequential()
    
    for i in range(num_layers):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=input_shape, kernel_initializer='he_normal'))
        else:
            model.add(tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu', kernel_initializer='he_normal'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
     
    return model