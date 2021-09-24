# Creating the dataset of specified directory
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


# Hutton class model
class Hutton_Dataset:
    def __init__(self):
        self.BATCH_SIZE = 4
        self.NUM_CLASSES = 4
        self.IMG_HEIGHT = 180
        self.IMG_WIDTH = 180
        self.DATA_DIR = None
        self.SOURCE_DATA_DIR = None
        self.TRAIN_DATASET = None
        self.VALIDATION_DATASET = None
        self.ACTIVE_DATA_DIR = None

    # Get the current training dataset
    def get_train_dataset(self):
        return self.TRAIN_DATASET

    # Prepares a training dataset from 80% of the images
    # and sets that to TRAIN_DATASET
    def prepare_train_dataset(self):
        if self.ACTIVE_DATA_DIR is None or self.BATCH_SIZE is None:
            return None
        if self.IMG_WIDTH is None or self.IMG_HEIGHT is None:
            return None

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.ACTIVE_DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE,
        )
        self.TRAIN_DATASET = train_ds

    # Get the current validation dataset
    def get_validation_dataset(self):
        return self.VALIDATION_DATASET

    # Prepares a training dataset from 20% of the images
    def prepare_validation_dataset(self):
        if self.ACTIVE_DATA_DIR is None or self.BATCH_SIZE is None:
            return None
        if self.IMG_WIDTH is None or self.IMG_HEIGHT is None:
            return None
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.ACTIVE_DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE
        )
        self.VALIDATION_DATASET = val_ds

    def create_batches(self):
        # To tune the values dynamically in runtime
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds = self.TRAIN_DATASET.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = self.VALIDATION_DATASET.cache().prefetch(buffer_size=AUTOTUNE)
        # Updating the current values of both train and validation datasets
        self.TRAIN_DATASET = train_ds
        self.VALIDATION_DATASET = val_ds
        # Standardize values in range [0,1] by using Rescaling layer
        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
        # Map dataset
        normalized_ds = self.TRAIN_DATASET.map(lambda x, y: (normalization_layer(x), y))
        images_batch, labels_batch = next(iter(normalized_ds))
        return images_batch, labels_batch

    # Gets the active data directory because
    # the client can define both absolute and external data sources.
    # Therefore, there must be a control element that the datasets will use
    def get_ACTIVE_DATA_DIR(self):
        return self.ACTIVE_DATA_DIR

    # Set the absolute data directory
    def set_ACTIVE_DATA_DIR(self, new_data_dir):
        self.ACTIVE_DATA_DIR = new_data_dir

    # Get the absolute data directory
    def get_DATA_DIR(self):
        return self.DATA_DIR

    # Set the absolute data directory
    def set_DATA_DIR(self, new_data_dir):
        self.DATA_DIR = new_data_dir

    def get_SOURCE_DATA_DIR(self):
        return self.SOURCE_DATA_DIR

    def set_SOURCE_DATA_DIR(self, file_name, file_url):
        raw_data_path = tf.keras.utils.get_file(
            file_name,
            file_url,
            extract=True)
        extracted_data_path = os.path.join(os.path.dirname(raw_data_path), 'train')
        self.SOURCE_DATA_DIR = extracted_data_path

    # Get number of classes or labels
    def get_NUM_CLASSES(self):
        return self.NUM_CLASSES

    # Set number of classes or labels
    def set_NUM_CLASSES(self, new_num_classes):
        self.NUM_CLASSES = new_num_classes

    # Get batch size to define how many batches will be created
    # to combine consecutive elements of the dataset
    def get_BATCH_SIZE(self):
        return self.BATCH_SIZE

    # Set batch size
    def set_BATCH_SIZE(self, new_batch_size):
        self.BATCH_SIZE = new_batch_size

    # Get the image height, which is same for all images
    def get_IMG_HEIGHT(self):
        return self.IMG_HEIGHT

    # Set the image height
    def set_IMG_HEIGHT(self, new_img_height):
        self.IMG_HEIGHT = new_img_height

    # Get the image width, which is same for all images
    def get_IMG_WIDTH(self):
        return self.IMG_WIDTH

    # Set the image width
    def set_IMG_WIDTH(self, new_img_width):
        self.IMG_WIDTH = new_img_width

    def info(self):
        # Get the latest data
        img_height = self.get_IMG_HEIGHT()
        img_width = self.get_IMG_WIDTH()
        batch_size = self.get_BATCH_SIZE()
        number_classes = self.get_NUM_CLASSES()
        current_train_ds = self.get_train_dataset()
        current_val_ds = self.get_validation_dataset()
        first_image = image_batch[0]
        # Configure the data for a better visualisation
        configuration = ""
        configuration += "Data directory: {}".format(active_data_dir) + "\n"
        configuration += "External data directory: {}".format(source_data_dir) + "\n"
        configuration += "Image height and width: {}x{}".format(img_height, img_width) + "\n"
        configuration += "Batch size: {}".format(batch_size) + "\n"
        configuration += "Number of classes(labels): {}".format(number_classes) + "\n"
        configuration += "Train dataset: {}".format(current_train_ds) + "\n"
        configuration += "Validation dataset: {}".format(current_val_ds) + "\n"
        configuration += "First mapped image in dataset: {}-{}".format(np.min(first_image), np.max(first_image))
        return configuration


# Prepare train and validation datasets to make Hutton model functional
hutton_v1_dataset = Hutton_Dataset()
# You can use your own data by just parameterizing the absolute directory
data_dir = "D:/GitRepos/hutton/rock_samples/train/"
hutton_v1_dataset.set_DATA_DIR(data_dir)
# You can also use external data sources that are zipped
external_data_url = 'https://github.com/Quelich/hutton/blob/main/rock_samples/train.zip'
hutton_v1_dataset.set_SOURCE_DATA_DIR('train.zip', external_data_url)
source_data_dir = hutton_v1_dataset.get_SOURCE_DATA_DIR()
# Set the active data source directory
active_data_dir = data_dir
hutton_v1_dataset.set_ACTIVE_DATA_DIR(active_data_dir)

# Set parameters
# Height of the each image in the dataset
hutton_v1_dataset.set_IMG_HEIGHT(new_img_height=180)
# Width of the each image in the dataset
hutton_v1_dataset.set_IMG_WIDTH(new_img_width=180)
# Number of labels in the dataset
hutton_v1_dataset.set_NUM_CLASSES(new_num_classes=4)
# Batch size that will be formed from the images
hutton_v1_dataset.set_BATCH_SIZE(new_batch_size=4)

# Prepare the training and validation datasets
hutton_v1_dataset.prepare_train_dataset()
hutton_v1_dataset.prepare_validation_dataset()
# Create batches
image_batch, label_batch = hutton_v1_dataset.create_batches()
# Display the parameters for better understanding
dataset_info = hutton_v1_dataset.info()
print(dataset_info)

