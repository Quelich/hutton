import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tensorflow.keras import layers
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

# Data PARAMETERS
BATCH_SIZE = 3
IMG_HEIGHT = 180
IMG_WIDTH = 180

data_dir = 'D:/GitRepos/hutton/rock_samples/train'

data = DataLoader.from_folder(data_dir)
train_data, test_data = data.split(0.9)
# Customize the TensorFlow model
hutton_tflite = image_classifier.create(train_data)
# Evaluate the model
loss, accuracy = hutton_tflite.evaluate(test_data)
# Quantization
config = QuantizationConfig.for_float16()
# # Export to TensorFlow Lite model
hutton_tflite.export(export_dir='D:/GitRepos/hutton/hutton_v1_tflite',
                     tflite_filename='hutton.tflite',
                     quantization_config=config)

hutton_tflite.export(export_dir='D:/GitRepos/hutton/hutton_v1_tflite',
                     export_format=ExportFormat.LABEL)

# # Training 80% of the images
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
# )
# # Validating 20% of the images
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     data_dir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE
# )
#
# class_names = train_ds.class_names
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# # Standardize values in range [0,1] by using Rescaling layer
# normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
#
# # Mapping
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#
# # Creating Eager tensors for easy debugging
# image_batch, labels_batch = next(iter(normalized_ds))
#
# # Concatenating the subtensors to visualize the process
# concatenated_tensor = tf.concat([image_batch[0], image_batch[1], image_batch[2]], 0)
#
# # Splitting the data into "train" and "test" categories
# train_data, test_data = tf.split(concatenated_tensor, num_or_size_splits=2, axis=1)
#
# train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
# test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)
# print("--------------Train Data--------------")
# print(train_data)
# print("--------------Test Data--------------")
# print(test_data)

