# [hutton](https://www.amnh.org/learn-teach/curriculum-collections/earth-inside-and-out/james-hutton)
This project aims to classify rock types using image classification methods powered by Tensorflow.
## Required Modules for Standalone Desktop Utilization
```Python
python pip install os
python pip install tensorflow
python pip install numpy
python pip install keras
```
## Prepare a Image Dataset 
-[Required] Prepare train and validation datasets to make Hutton model 
```Python
# Initialize the dataset instance
hutton_v1_dataset = Hutton_Dataset()
# You can use your own data by just parameterizing the absolute directory
data_dir = "D:/GitRepos/hutton/rock_samples/train"
hutton_v1_dataset.set_DATA_DIR(data_dir)
# You can also use external data sources that are zipped
external_data_url = 'https://github.com/Quelich/hutton/blob/main/rock_samples/train.zip'
hutton_v1_dataset.set_SOURCE_DATA_DIR('train.zip', external_data_url)
source_data_dir = hutton_v1_dataset.get_SOURCE_DATA_DIR()
hutton_v1_dataset.prepare_train_dataset()
# hutton_v1.get_train_dataset()
hutton_v1_dataset.prepare_validation_dataset()
hutton_v1_dataset.autotune_datasets()
image_batch, label_batch = hutton_v1_dataset.create_batches()
```
- [Optional] Set up the parameters
```Python
img_height = hutton_v1_dataset.get_IMG_HEIGHT()
img_width = hutton_v1_dataset.get_IMG_WIDTH()
batch_size = hutton_v1_dataset.get_BATCH_SIZE()
number_classes = hutton_v1_dataset.get_NUM_CLASSES()
current_train_ds = hutton_v1_dataset.get_train_dataset()
current_val_ds = hutton_v1_dataset.get_validation_dataset()
first_image = image_batch[0]
```
- [Optional] Display the parameters to visualize the infrastructure
```Python
print("Data directory: {}".format(data_dir))
print("External data directory: {}".format(source_data_dir))
print("Image height and width: {}x{}".format(img_height,img_width))
print("Batch size: {}".format(batch_size))
print("Number of classes(labels): {}".format(batch_size))
print("Train dataset: {}".format(current_train_ds))
print("Validation dataset: {}".format(current_val_ds))
print("First mapped image in dataset: {}-{}".format(np.min(first_image), np.max(first_image)))
```
