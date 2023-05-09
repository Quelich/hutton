# Creating the dataset of specified directory
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from hutton_v1.utilities.hutton_utilities import _visualizeData_
from hutton_v1.utilities.hutton_utilities import _logResults_
from hutton_v1.utilities.hutton_utilities import _getResults_


is_visualize = False
# Data PARAMETERS TODO optimize parameters
BATCH_SIZE = 4
IMG_HEIGHT = 180
IMG_WIDTH = 180
NUM_CLASSES = 4

# Local directory of the data
data_dir = 'C:/repos/hutton/rock_samples/train'


# Training 80% of the images
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

# Validating 20% of the images
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Displaying the batch information
# print("Train dataset batches")
# _retrieveBatches_(train_ds)
# print("Validation dataset batches")
# _retrieveBatches_(val_ds)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize values in range [0,1] by using Rescaling layer
normalization_layer = layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# Tensors
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# # Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# ML MODEL ! --------------------------------------------------MODEL
# ----------------------------------------------------------------------- TODO Initialize Report File (PDF?) every
#  process should be documented into a report file


# The machine learning model named in honor of James Hutton
# TODO optimize the model
Hutton = Sequential([
    layers.Rescaling(
        1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(2, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(4, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(4, activation='relu',
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(NUM_CLASSES)
])

# Model Optimization
adam = tf.keras.optimizers.Adam(learning_rate=0.00008,
                                beta_1=0.9,
                                beta_2=0.999,
                                amsgrad=False)

# Model Compilation
Hutton.compile(
    optimizer=adam,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Model Summary
Hutton.summary()

# Model Training
epochs = 25
history = Hutton.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualize training results
if is_visualize:
    _visualizeData_(history, epochs)
    plt.show()

# Fixing Overfitting - Data Augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(IMG_HEIGHT,
                                       IMG_WIDTH,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Recompile the model Hutton
Hutton.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(
                   from_logits=True),
               metrics=['accuracy'])

Hutton.summary()

history = Hutton.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
# Visualize training results
if is_visualize:
    _visualizeData_(history, epochs)

test_dir = "C:/repos/hutton/rock_samples/test/coal/1.jpg"

# TODO use for loop for multiple image files
test_img = tf.keras.utils.load_img(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH)
)

img_array = tf.keras.utils.img_to_array(test_img)
# Create a batch
img_array = tf.expand_dims(img_array, 0)

predictions = Hutton.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# Displaying the results

results = _getResults_(class_names, score)
print(results)

# Logging the results
output_data = results + ";for " + test_dir

_logResults_(output_data)


# TODO add this method to the class

def hutton_classification_result():
    classification_result = _getResults_(class_names, score)
    return classification_result
