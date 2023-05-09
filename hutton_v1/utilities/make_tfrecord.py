import tensorflow as tf
from keras import layers
import numpy as np
from IPython.display import display

# Data PARAMETERS
BATCH_SIZE = 3
IMG_HEIGHT = 180
IMG_WIDTH = 180
# Local directory of the data
data_dir = 'D:/repos/hutton/rock_samples/train'

# Training 80% of the images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
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

class_names = train_ds.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize values in range [0,1] by using Rescaling layer
normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# Creating Tensors
image_batch, labels_batch = next(iter(normalized_ds))

# Float feature
feature0 = image_batch[0]
# Float feature
feature1 = image_batch[1]
# Float feature
feature2 = image_batch[2]


# TODO send to hutton_utilities
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2):
    """
  Creates a tf.train.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'feature0': _float_feature(feature0),
        'feature1': _float_feature(feature1),
        'feature2': _float_feature(feature2),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(f0, f1, f2):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2),  # Pass these args to the above function.
        tf.string)  # The return type is `tf.string`.
    return tf.reshape(tf_string, ())  # The result is a scalar.


# Writing a TFRecord file
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2))
# print(features_dataset)

# Use `take(1)` to only pull one example from the dataset.
for f0,f1,f2 in features_dataset.take(1):
  print(f0)

print(tf_serialize_example(f0, f1, f2))