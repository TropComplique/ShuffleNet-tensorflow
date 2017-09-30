# ShuffleNet

## Summary
This is an implementation of
[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083).
It is written in Tensorflow and tested on [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset.
The dataset consists of 64x64 images and has 200 classes.

## Implementation details
* For data augmentation I use 56x56 random crops and random color manipulations.
* I use a reduce-on-plateau learning rate scheduler.
* I use slightly reduced in size ShuffleNet: in the original paper it has more layers.
* For the input pipeline I use `tf.contrib.data.TFRecordDataset`.

## How to use it
* Convert your dataset to `tfrecords` format by using `image_dataset_to_tfrecords.py`.
* If you want, edit number of various layers in `shufflenet/architecture.py`.
* Run `train.py`.

## Requirements
* Python 3.6
* tensorflow 1.3
* tqdm, Pillow, pandas, numpy
