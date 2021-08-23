# Creating the dataset of specified directory
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, experimental
from tensorflow.keras import Model

# Hutton class model
class Hutton(Model):
    def __init__(self, batch_size, num_classes, img_height, img_width):
        super(Hutton, self).__init__()
        self.BATCH_SIZE = batch_size
        self.NUM_CLASSES = num_classes
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.preprocessing = experimental.preprocessing.Rescaling(1. / 255,
                                                                  input_shape=(img_height, img_width, 3))
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
        self.d2 = Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass

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