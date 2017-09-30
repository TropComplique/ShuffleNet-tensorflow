import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import os
import io
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_dir', type=str,
    default='/home/ubuntu/data/tiny-imagenet-200/training',
    help='A path to a folder with training data.'
)
parser.add_argument(
    '--val_dir', type=str,
    default='/home/ubuntu/data/tiny-imagenet-200/validation',
    help='A path to a folder with validation data.'
)
parser.add_argument(
    '--save_dir', type=str,
    default='/home/ubuntu/data/tiny-imagenet-200',
    help='A path to a folder where to save results.'
)
args = parser.parse_args()


"""The purpose of this script is
to convert image dataset that looks like:
    class1/image1.jpg
    class1/image44.jpg
    class1/image546.jpg
    ...
    class6/image55.jpg
    class6/image12.jpg
    class6/image76.jpg
    ...
to tfrecords format.

1. It assumes that each folder is separate class and
that the number of classes equals to the number of folders.

2. Also it assumes that validation and training folders
have the same subfolders (the same classes).

3. Additionally it outputs 'class_encoder.npy' file
that contains dictionary: folder_name -> class_index (integer).
"""


def main():
    encoder = create_encoder(args.train_dir)
    # now you can get a folder's name from a class index

    np.save(os.path.join(args.save_dir, 'class_encoder.npy'), encoder)
    convert(args.train_dir, encoder, os.path.join(args.save_dir, 'train.tfrecords'))
    convert(args.val_dir, encoder, os.path.join(args.save_dir, 'val.tfrecords'))

    print('\nCreated two tfrecords files:')
    print(os.path.join(args.save_dir, 'train.tfrecords'))
    print(os.path.join(args.save_dir, 'val.tfrecords'))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# here you can also just use `return array.tostring()`
# but it will make tfrecords file large
def to_bytes(array):
    image = Image.fromarray(array)
    tmp = io.BytesIO()
    image.save(tmp, format='jpeg')
    return tmp.getvalue()


def convert(folder, encoder, tfrecords_filename):
    """Convert a folder with directories of images to tfrecords format.

    Arguments:
        folder: A path to a folder where directories with images are.
        encoder: A dict, folder_name -> integer.
        tfrecords_filename: A path where to save tfrecords file.
    """

    images_metadata = collect_metadata(folder, encoder)
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
        target = int(row.class_number)

        feature = {
            'image': _bytes_feature(to_bytes(array)),
            'target': _int64_feature(target),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def create_encoder(folder):
    """Encode directories in the folder with integer values.
    Values are in the range 0..(n_directories - 1).

    Arguments:
        folder: A path to a folder where directories with images are.
            Each directory - separate class.
    Returns:
        A dict.
    """
    classes = os.listdir(folder)
    encoder = {n: i for i, n in enumerate(classes)}
    return encoder


def collect_metadata(folder, encoder):
    """Collect paths to images. Collect their classes.
    All paths must be with respect to 'folder'.

    Arguments:
        folder: A path to a folder where directories with images are.
            Each directory - separate class.
        encoder: A dict, folder_name -> integer.
    Returns:
        A pandas dataframe.
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

    # encode folder names by integers
    M['class_number'] = M.class_name.apply(lambda x: encoder[x])

    # shuffle the dataframe
    M = M.sample(frac=1).reset_index(drop=True)

    return M


if __name__ == '__main__':
    main()
