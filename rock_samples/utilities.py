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
