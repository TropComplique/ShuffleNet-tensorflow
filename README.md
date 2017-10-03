# ShuffleNet

This is an implementation of
[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083).
It is written in Tensorflow and tested on [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset.
The dataset consists of 64x64 images and has 200 classes. The network gives validation accuracy ~49% after 40 epochs (it takes ~2.2 hours on p2.xlarge).

## Implementation details
* I use reduced in size ShuffleNet: in the original paper it has more layers.  
But it is easy to change the code in `shufflenet/architecture.py` to make it like the original.
* For the input pipeline I use `tf.contrib.data.TFRecordDataset`.
* For data augmentation I use 56x56 sized random crops and random color manipulations.
* I use a reduce-on-plateau learning rate scheduler.

## How to use it
Assuming that Tiny ImageNet data is in `/home/ubuntu/data/tiny-imagenet-200/` steps are
* `cd ShuffleNet-tensorflow`.
* `python tiny_imagenet/move_data.py`  
to slightly change the folder structure of the data.
* `python image_dataset_to_tfrecords.py`  
to convert the dataset to `tfrecords` format.
* (optional) If you want to change the network's length,  
edit the number of ShuffleNet Units in `shufflenet/architecture.py`.
* `python train.py`  
to begin training. Evaluation is after each epoch.
* logs and the saved model will be in `logs/run0` and `saved/run0`.

To train on your dataset, you need to change a couple of parameters in the code.

## Requirements
* Python 3.6
* tensorflow 1.3
* tqdm, Pillow, pandas, numpy
