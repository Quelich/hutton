# [hutton](https://www.amnh.org/learn-teach/curriculum-collections/earth-inside-and-out/james-hutton)
This project aims to classify rock types using image classification methods powered by Tensorflow.
## Infrastructure
- Python 3.7
## Required Modules for Standalone Desktop Utilization
```Python
python pip install os
python pip install tensorflow
python pip install numpy
python pip install keras
```
## Fundamental Use of Hutton Pipeline
### _Prepare a Image Dataset for Hutton_
- [Required] Prepare train and validation datasets to make Hutton model 
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
# Set the active data source directory
active_data_dir = data_dir
hutton_v1_dataset.set_ACTIVE_DATA_DIR(active_data_dir)
# Prepare the training and validation datasets
hutton_v1_dataset.prepare_train_dataset()
hutton_v1_dataset.prepare_validation_dataset()
# Create batches
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
### _Instantiate Hutton Image Classifier_ 
## Desktop
_coming soon_
## Android
_coming soon_
