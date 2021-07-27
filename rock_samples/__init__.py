# Creating the dataset of specified directory
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.metrics import acc

from utilities import _getClasses_
from utilities import _plotImages_
from utilities import _retrieveBatches_
from utilities import _visualizeData_
from utilities import _visualizeAugmentedData_
# Data PARAMETERS
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
# Data directory
data_dir = 'D:/GitRepos/hutton/rock_samples/train/'

# Training 80% of the images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)
# Validating 20% of the images
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Visualizing the data
# _plotImages_(train_ds)
# print("Train dataset batches")
# _retrieveBatches_(train_ds)
# print("Validation dataset batches")
# _retrieveBatches_(val_ds)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize values in range [0,1] by using Rescaling layer
normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# ML MODEL

NUM_CLASSES = 1

# The machine learning model named in honour of James Hutton
hutton = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(1, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(2, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # layers.Conv2D(4, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(2, activation='relu'),
    layers.Dense(NUM_CLASSES)
])

# Model Compilation

hutton.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Model Summary
hutton.summary()

# Model Training
epochs = 15
history = hutton.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualize training results
# _visualizeData_(history, epochs)
# plt.show()

# An profound approach to Overfitting - Data Augmentation
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(IMG_HEIGHT,
                                                                  IMG_WIDTH,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)
# Visualizing the augmented images
# _visualizeAugmentedData_(train_ds, data_augmentation)

# Compile the model Hutton
hutton.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

hutton.summary()

epochs = 15
history = hutton.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
# Visualize training results
_visualizeData_(history, epochs)
