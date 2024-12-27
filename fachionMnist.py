import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras import layers

# Cargar el conjunto de datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalizar los datos
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Verificar las dimensiones
print(f'Tamaño del conjunto de entrenamiento: {x_train.shape}')
print(f'Tamaño del conjunto de prueba: {x_test.shape}')

# Ajustar la forma de los datos para que tengan 1 canal (escala de grises)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Construir el modelo
model = keras.Sequential([
    # Primera capa de convolución
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),

    # Segunda capa de convolución
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Tercera capa de convolución
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    # Cuarta capa de convolución
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Aplanamiento
    layers.Flatten(),

    # Primera capa densa
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Segunda capa densa
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Capa de salida
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Mensaje final para confirmar que el script se ha ejecutado completamente
print("El modelo se ha construido y compilado correctamente.")

# Ajustar la forma de los datos para que tengan 1 canal (escala de grises)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Precisión del modelo en el conjunto de prueba: {accuracy:.2f}')

# Graficar la pérdida
plt.figure(figsize=(12, 4))

# Pérdida de entrenamiento
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Precisión de entrenamiento
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()  # Ajustar el espaciado entre subgráficas
plt.show()  