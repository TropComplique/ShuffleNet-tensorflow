import tensorflow as tf


def _parse_and_preprocess(example_proto, image_size):

    features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'target': tf.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
    target = tf.cast(parsed_features['target'], tf.int32)

    image_shape = tf.stack([image_size, image_size, 3])
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32)

    return image, target


def _get_data(num_classes, image_size, is_training):

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

    train_dataset = train_dataset.map(lambda x: _parse_and_preprocess(x, image_size))
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat()

    val_dataset = val_dataset.map(lambda x: _parse_and_preprocess(x, image_size))
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

    return init_data, train_init, val_init, x_batch, y_batch
