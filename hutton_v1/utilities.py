# TODO _logResults_ needs to process all the data details
# TODO the resultant data might be serialized as JSON file
import os
import matplotlib.pyplot as plt
import datetime
import glob
from PIL import Image
import imghdr





def _addImages_(file_dir):
    data_dirs = []
    for filename in os.listdir(file_dir):
        if filename.endswith('.jpg'):
            data_dirs.append(os.path.join(file_dir, filename))
            print("added {}".format(os.path.join(file_dir, filename)))
    return data_dirs


def _plotImages_(train_dataset):
    class_names = train_dataset.class_names
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
    batch_data = ""
    for image_batch, labels_batch in train_dataset:
        batch_info = "Image Batch: {} Labels Batch: {}\n".format(image_batch.shape, labels_batch.shape)
        batch_data += batch_info
    return batch_data  # Indicates the process is done


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
            plt.title('Augmented Data')
            plt.axis("off")

    plt.show()


def _getToday_():
    # today = datetime.date.today().strftime("%B  %d, %Y")
    today = datetime.date.today()
    return today


def _logResults_(output_data):
    # Relative log file path
    logfile_relative_path = "result_logs.txt"
    # Absolute log file path
    logfile_absolute_path = 'D:/GitRepos/hutton/hutton_v1/' + logfile_relative_path
    # Checking if the log file exists
    is_exist = os.path.exists(logfile_absolute_path)
    # Gets the date of program execution, necessary for logging
    today_date = _getToday_()
    # If not, creates a new log file
    if not is_exist:
        logfile = open(logfile_relative_path, "w")
        logfile.write("Log file is initialized at {}".format(str(today_date)))
        logfile.close()
    # Logs the the results with date
    with open(logfile_relative_path, "a") as logs:
        logs.write("Recorded at {}; {}\n".format(str(today_date), output_data))


def _getLogResults_():
    # Relative log file path
    logfile_relative_path = "result_logs.txt"
    # Absolute log file path
    logfile_absolute_path = 'D:/GitRepos/hutton/hutton_v1/' + logfile_relative_path
    # Checking if the log file exists
    is_exist = os.path.exists(logfile_absolute_path)
    # If not, returns nothing
    if not is_exist:
        raise ValueError("NULL Log file ")
    with open(logfile_relative_path, "r") as logs:
        print("---------------------------------------------------------------------")
        print(logs.read())
        print("---------------------------------------------------------------------")


def _enumerateImagesDir_(data_dir):
    i = 1
    for file in os.listdir(data_dir):
        # print(data_dir + "/" + file)
        os.rename(data_dir + "/" + file, data_dir + "/" + str(i) + ".jpg")
        i = i + 1


# def _fixImageChannels_(data_dir):
#     print(data_dir)
#     for file in os.listdir(data_dir):
#         image = cv2.imread(file)
#         file_type = imghdr.what(file)
#         if file_type != 'jpeg':
#             print(file + " - invalid - " + str(file_type))
#     #  cv2.imwrite(file, image)
