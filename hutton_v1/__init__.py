# Creating the dataset of specified directory
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utilities import _plotImages_
from utilities import _retrieveBatches_
from utilities import _visualizeData_
from utilities import _visualizeAugmentedData_
from utilities import _logResults_

# Data PARAMETERS
BATCH_SIZE = 2
IMG_HEIGHT = 180
IMG_WIDTH = 180
# Data directory
data_dir = 'D:/GitRepos/hutton/rock_samples/train'

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

# Displaying the batch information
print("Train dataset batches")
_retrieveBatches_(train_ds)
print("Validation dataset batches")
_retrieveBatches_(val_ds)

class_names = train_ds.class_names
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

NUM_CLASSES = 2

# The machine learning model named in honour of James Hutton
hutton = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(2, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(4, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
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

hutton.compile(
    optimizer=adam,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Model Summary
hutton.summary()

# Model Training
epochs = 25
history = hutton.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Visualize training results
_visualizeData_(history, epochs)
plt.show()

# Fixing Overfitting - Data Augmentation
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
_visualizeAugmentedData_(train_ds, data_augmentation)

# Compile the model Hutton
hutton.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

hutton.summary()


history = hutton.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
# Visualize training results
_visualizeData_(history, epochs)

test_dir = 'D:/GitRepos/hutton/rock_samples/test/bst_5.jpg'

test_img = keras.preprocessing.image.load_img(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH)
)

img_array = keras.preprocessing.image.img_to_array(test_img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = hutton.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Displaying the results
results = "This image most likely belongs to {} with a {:.2f} percent confidence".format(class_names[np.argmax(score)],
                                                                                          100 * np.max(score))
print(results)

# Logging the results
output_data = results + ";for " + test_dir

_logResults_(output_data)
