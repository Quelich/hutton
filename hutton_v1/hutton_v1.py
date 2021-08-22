# Creating the dataset of specified directory
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from hutton_utilities import _visualizeData_
from hutton_utilities import _visualizeAugmentedData_
from hutton_utilities import _logResults_
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, experimental
from tensorflow.keras import Model

class Hutton(Model):
    BATCH_SIZE = 3
    NUM_CLASSES = 3
    IMG_HEIGHT = 180
    IMG_WIDTH = 180

    def __init__(self, batch_size, num_classes, img_height, img_width):
        super(Hutton, self).__init__()
        self.BATCH_SIZE = batch_size
        self.NUM_CLASSES = num_classes
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.preprocessing = experimental.preprocessing.Rescaling(1. / 255,
                                                                  input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))
        self.conv1 = Conv2D(2, 3, padding='same', activation='relu')
        self.pooling1 = MaxPooling2D()
        self.conv2 = Conv2D(4, 3, padding='same', activation='relu')
        self.pooling2 = MaxPooling2D()
        self.dropout1 = Dropout(0.1)
        self.conv3 = Conv2D(8, 3, padding='same', activation='relu')
        self.pooling3 = MaxPooling2D()
        self.dropout2 = Dropout(0.2)
        self.flatten = Flatten()
        self.d1 = Dense(4, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.d2 = Dense(self.NUM_CLASSES)

    def get_NUM_CLASSES(self):
        return self.NUM_CLASSES

    def set_NUM_CLASSES(self, new_class_value):
        self.NUM_CLASSES = new_class_value

    def get_BATCH_SIZE(self):
        return self.BATCH_SIZE

    def set_BATCH_SIZE(self, new_batch_size):
        self.BATCH_SIZE = new_batch_size

    def get_IMG_HEIGHT(self):
        return self.IMG_HEIGHT

    def set_IMG_HEIGHT(self, new_img_height):
        self.IMG_HEIGHT = new_img_height

    def get_IMG_WIDTH(self):
        return self.IMG_WIDTH

    def set_IMG_WIDTH(self, new_img_width):
        self.IMG_WIDTH = new_img_width

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass
