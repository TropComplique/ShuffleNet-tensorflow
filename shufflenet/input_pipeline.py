import tensorflow as tf


def _get_data(num_classes, image_size):

    batch_size = tf.Variable(
        tf.placeholder(tf.int64, [], 'batch_size'),
        trainable=False, collections=[]
    )
    train_file = tf.Variable(
        tf.placeholder(tf.string, [], 'train_file'),
        trainable=False, collections=[]
    )
    val_file = tf.Variable(
        tf.placeholder(tf.string, [], 'val_file'),
        trainable=False, collections=[]
    )
    init_data = tf.variables_initializer([batch_size, train_file, val_file])

    train_dataset = tf.contrib.data.TFRecordDataset(train_file)
    val_dataset = tf.contrib.data.TFRecordDataset(val_file)

    train_dataset = train_dataset.map(
        lambda x: _parse_and_preprocess(x, image_size, augmentation=True),
        num_threads=4,
        output_buffer_size=1000
    )

    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat()

    val_dataset = val_dataset.map(
        lambda x: _parse_and_preprocess(x, image_size)
    )
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.repeat()

    iterator = tf.contrib.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
    )

    train_init = iterator.make_initializer(train_dataset)
    val_init = iterator.make_initializer(val_dataset)

    x_batch, y_batch = iterator.get_next()
    y_batch = tf.one_hot(y_batch, num_classes, axis=1, dtype=tf.float32)

    data = {
        'init_data': init_data,
        'train_init': train_init, 'val_init': val_init,
        'x_batch': x_batch, 'y_batch': y_batch
    }
    return data


def _parse_and_preprocess(example_proto, image_size, augmentation=False):

    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    target = parsed_features['target']

    if augmentation:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.8, 1.25)
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_saturation(image, 0.8, 1.25)
        image = tf.random_crop(image, [image_size, image_size, 3])
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, target
