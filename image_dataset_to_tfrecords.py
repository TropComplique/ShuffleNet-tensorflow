import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import os


def create_encoder(folder):
    classes = os.listdir(folder)
    encoder = {n: i for i, n in enumerate(classes)}
    return encoder


def collect_metadata(folder, encoder):
    """Collect paths to images. Collect their classes.

    Arguments:
        folder: A path to a folder where directories with images are.
            Each directory - separate class.
            Name of a directory - number of class.
            Numbering is from 1 to n_classes.
    Returns:
        M: A pandas dataframe.
    """

    subdirs = list(os.walk(folder))[1:]
    metadata = []

    for dir_path, _, files in subdirs:
        dir_name = dir_path.split('/')[-1]
        for file_name in files:
            image_metadata = [dir_name, os.path.join(dir_name, file_name)]
            metadata.append(image_metadata)

    M = pd.DataFrame(metadata)
    M.columns = ['class_name', 'img_path']

    M['class_number'] = M.class_name.apply(lambda x: encoder[x])

    # shuffle the dataframe
    M = M.sample(frac=1).reset_index(drop=True)

    return M


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert(images_metadata, folder, tfrecords_filename):
    """Convert a folder with directories of images to tfrecords format.

    Arguments:
        images_metadata: A pandas dataframe that contains paths to images
            and classes of these images.
            It must contain columns 'img_path' and 'class_number'.
            All paths must be with respect to 'folder'.
        folder: A path to a folder where directories with images are.
        tfrecords_filename: A path where to save tfrecords file.
    """

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for _, row in tqdm(images_metadata.iterrows()):

        file_path = os.path.join(folder, row.img_path)
        # read an image
        image = Image.open(file_path)
        # convert to an array
        array = np.asarray(image, dtype='uint8')
        
        # some images are grayscale
        if array.shape[-1] != 3:
            array = np.stack([array, array, array], axis=2)
        
        # get class of the image
        target = row.class_number

        feature = {
            'image_raw': _bytes_feature(array.tostring()),
            'target': _int64_feature(target)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


data_dir = '/home/ubuntu/data/tiny-imagenet-200/'

train_dir = data_dir + 'training'
val_dir = data_dir + 'validation'

encoder = create_encoder(train_dir)
np.save('encoder.npy', encoder)

train_metadata = collect_metadata(train_dir, encoder)
val_metadata = collect_metadata(val_dir, encoder)

# warning: tfrecords files can be big
convert(train_metadata, train_dir, data_dir + 'train.tfrecords')
convert(val_metadata, val_dir, data_dir + 'val.tfrecords')
