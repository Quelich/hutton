import os
import matplotlib.pyplot as plt


# Adding the image directories to a list
def _addImages_(file_dir):
    data_dirs = []
    for filename in os.listdir(file_dir):
        if filename.endswith('.jpg'):
            data_dirs.append(os.path.join(file_dir, filename))
            print("added {}".format(os.path.join(file_dir, filename)))
    return data_dirs


def _getClasses_(train_dataset):
    class_names = train_dataset.class_names
    return class_names


def _plotImages_(train_dataset):
    class_names = _getClasses_(train_dataset)
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for image in range(9):  # size the batch up
            ax = plt.subplot(3, 3, image + 1)
            # Converting tensors to Numpy array
            plt.imshow(images[image].numpy().astype("uint8"))
            plt.title(class_names[labels[image]])
            plt.axis("off")
    plt.show()


def _retrieveBatches_(train_dataset):
    for image_batch, labels_batch in train_dataset:
        print("Image Batch: {}\n"
              "Labels Batch: {}".format(image_batch.shape, labels_batch.shape))
    return True  # Indicates the process is done


def _visualizeData_(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def _visualizeAugmentedData_(train_ds, data_augmentation):
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()