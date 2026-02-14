#############################################
# Lesson 121 - Introduction to CNN
# Activity 1: Train your model for images
# ACP: Applications of CNN
#############################################

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("Lesson 121: Training a basic image model (CIFAR-10)")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model_121 = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_121.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_121.summary()
model_121.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

#############################################
# Lesson 122 - Layers in CNN
# Activity 1: CNN using Keras
# ACP: Operations in convolutional networks
#############################################

print("Lesson 122: CNN Layers Example")

model_122 = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_122.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_122.summary()

#############################################
# Lesson 123 - Image Classifier Using CNN Part 1
# Activity 1: Image Classifier Part 1
# ACP: Cat or Dog – Part 1
#############################################

print("Lesson 123: Cat vs Dog CNN Part 1")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    '/content/data/train/',
    target_size=(150,150),
    class_mode='binary'
)

model_123 = tf.keras.models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_123.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_123.fit(train_generator, epochs=3)

#############################################
# Lesson 124 - Image Classifier Using CNN Part 2
# Activity 1: Image Classifier Part 2
# ACP: Cat or Dog – Part 2
#############################################

print("Lesson 124: Cat vs Dog Part 2 (Augmentation)")

aug_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True
)

aug_generator = aug_datagen.flow_from_directory(
    '/content/data/train/',
    target_size=(150,150),
    class_mode='binary'
)

model_124 = model_123  # reuse architecture
model_124.fit(aug_generator, epochs=5)

#############################################
# Lesson 125 - Digit Recognizer using CNN – 1
# Activity 1: CNN for Fashion MNIST – 1
#############################################

print("Lesson 125: Fashion MNIST Part 1")

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train_f, y_train_f), (x_test_f, y_test_f) = fashion_mnist.load_data()

x_train_f = x_train_f.reshape(-1,28,28,1)/255.0
x_test_f  = x_test_f.reshape(-1,28,28,1)/255.0

model_125 = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_125.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_125.fit(x_train_f, y_train_f, epochs=5, validation_data=(x_test_f, y_test_f))

#############################################
# Lesson 126 - Digit Recognizer using CNN – 2
# Activity 1: CNN for Fashion MNIST – 2
#############################################

print("Lesson 126: Fashion MNIST Part 2 (Improved)")

model_126 = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

model_126.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_126.summary()
