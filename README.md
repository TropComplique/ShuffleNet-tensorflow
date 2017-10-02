# ShuffleNet

This is an implementation of
[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083).
It is written in Tensorflow and tested on [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset.
The dataset consists of 64x64 images and has 200 classes. The network gives accuracy ~..% after .. epochs (~.. hours on p2.xlarge) (See `logs` for configs)

## Implementation details
* I use slightly reduced in size ShuffleNet: in the original paper it has more layers. But it is easy to change the code in `shufflenet/architecture.py` to make it like the original.
* For the input pipeline I use `tf.contrib.data.TFRecordDataset`.
* For data augmentation I use 56x56 sized random crops and random color manipulations.
* I use a reduce-on-plateau learning rate scheduler.

## How to use it
Assuming that Tiny ImageNet data is in `/home/ubuntu/data/tiny-imagenet-200/` steps are
* `cd ShuffleNet-tensorflow`
* `python tiny_imagenet/move_data.py`  
to move the data a little bit.
* `python image_dataset_to_tfrecords.py`  
to convert the dataset to `tfrecords` format.
* (optional) If you want, edit numbers of various layers in `shufflenet/architecture.py`.
* `python train.py`  
to begin training. Evaluation is after each epoch.

## Requirements
* Python 3.6
* tensorflow 1.3
* tqdm, Pillow, pandas, numpy
