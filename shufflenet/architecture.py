import tensorflow as tf
import math


BATCH_NORM_MOMENTUM = 0.1
# it differs from the default value,
# sometimes right momentum value is very important


N_SHUFFLE_UNITS = [1, 3, 1]
# number of shuffle units of stride 1 in each stage,
# in the original paper: [3, 7, 3]

# number of layers that the network will have:
# (sum(N_SHUFFLE_UNITS) + 3)*3 + 1 + 1

# if input image has spatial size [56, 56]
# then spatial size before global average pooling is [4, 4]


def _channel_shuffle(X, groups):
    height, width, in_channels = X.shape.as_list()[1:]
    in_channels_per_group = int(in_channels/groups)

    # reshape
    shape = tf.stack([-1, height, width, groups, in_channels_per_group])
    X = tf.reshape(X, shape)

    # transpose
    X = tf.transpose(X, [0, 1, 2, 4, 3])

    # reshape
    shape = tf.stack([-1, height, width, in_channels])
    X = tf.reshape(X, shape)

    return X


# a ShuffleNet implementation
def _mapping(
        X, is_training, num_classes=200,
        groups=3, dropout=0.5,
        complexity_scale_factor=0.75
        ):

    # second stage's number of output channels
    if groups == 1:
        out_channels = 144
    elif groups == 2:
        out_channels = 200
    elif groups == 3:
        out_channels = 240
    elif groups == 4:
        out_channels = 272
    elif groups == 8:
        out_channels = 384
    # all 'out_channels' are divisible by corresponding 'groups'

    # possibly decrease network's width
    out_channels = int(out_channels * complexity_scale_factor)

    with tf.variable_scope('features'):

        with tf.variable_scope('stage1'):

            with tf.variable_scope('conv1'):
                result = _conv(X, 24, kernel=3, stride=1)
                # in the original paper they are using stride=2
                # but because I use small 64x64 images I chose stride=1

            result = _batch_norm(result, is_training)
            result = _nonlinearity(result)
            # in the original paper they are not using batch_norm and relu here

            result = _max_pooling(result)

        with tf.variable_scope('stage2'):

            with tf.variable_scope('unit1'):
                result = _first_shufflenet_unit(
                    result, is_training, groups, out_channels
                )

            for i in range(N_SHUFFLE_UNITS[0]):
                with tf.variable_scope('unit' + str(i + 2)):
                    result = _shufflenet_unit(result, is_training, groups)

            # number of channels in 'result' is out_channels

        with tf.variable_scope('stage3'):

            with tf.variable_scope('unit1'):
                result = _shufflenet_unit(result, is_training, groups, stride=2)

            for i in range(N_SHUFFLE_UNITS[1]):
                with tf.variable_scope('unit' + str(i + 2)):
                    result = _shufflenet_unit(result, is_training, groups)

            # number of channels in 'result' is 2*out_channels

        with tf.variable_scope('stage4'):

            with tf.variable_scope('unit1'):
                result = _shufflenet_unit(result, is_training, groups, stride=2)

            for i in range(N_SHUFFLE_UNITS[2]):
                with tf.variable_scope('unit' + str(i + 2)):
                    result = _shufflenet_unit(result, is_training, groups)

            # number of channels in 'result' is 4*out_channels

    with tf.variable_scope('classifier'):
        result = _global_average_pooling(result)

        result = _dropout(result, is_training, dropout)
        # in the original paper they are not using dropout here

        logits = _affine(result, num_classes)

    return logits


def _nonlinearity(X):
    return tf.nn.relu(X, name='ReLU')


def _dropout(X, is_training, rate=0.5):
    keep_prob = tf.constant(
        1.0 - rate, tf.float32,
        [], 'keep_prob'
    )
    result = tf.cond(
        is_training,
        lambda: tf.nn.dropout(X, keep_prob),
        lambda: tf.identity(X),
        name='dropout'
    )
    return result


def _batch_norm(X, is_training):
    return tf.layers.batch_normalization(
        X, scale=False, center=True,
        momentum=BATCH_NORM_MOMENTUM,
        training=is_training, fused=True
    )


def _global_average_pooling(X):
    return tf.reduce_mean(
        X, axis=[1, 2],
        name='global_average_pooling'
    )


def _max_pooling(X):
    return tf.nn.max_pool(
        X, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME',
        name='max_pooling'
    )


def _avg_pooling(X):
    return tf.nn.avg_pool(
        X, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME',
        name='avg_pooling'
    )


def _conv(X, filters, kernel=3, stride=1):

    in_channels = X.shape.as_list()[-1]

    # kaiming uniform initialization
    maxval = math.sqrt(6.0/in_channels)

    K = tf.get_variable(
        'kernel', [kernel, kernel, in_channels, filters],
        tf.float32, tf.random_uniform_initializer(-maxval, maxval)
    )

    b = tf.get_variable(
        'bias', [filters], tf.float32,
        tf.zeros_initializer()
    )

    return tf.nn.bias_add(
        tf.nn.conv2d(X, K, [1, stride, stride, 1], 'SAME'), b
    )


def _group_conv(X, filters, groups, kernel=1, stride=1):

    in_channels = X.shape.as_list()[3]
    in_channels_per_group = int(in_channels/groups)
    filters_per_group = int(filters/groups)

    # kaiming uniform initialization
    maxval = math.sqrt(6.0/in_channels_per_group)

    K = tf.get_variable(
        'kernel', [kernel, kernel, in_channels_per_group, filters],
        tf.float32, tf.random_uniform_initializer(-maxval, maxval)
    )

    # split channels
    X_channel_splits = tf.split(X, [in_channels_per_group]*groups, axis=3)
    K_filter_splits = tf.split(K, [filters_per_group]*groups, axis=3)

    results = []

    # do convolution for each split
    for i in range(groups):
        X_split = X_channel_splits[i]
        K_split = K_filter_splits[i]
        results += [tf.nn.conv2d(X_split, K_split, [1, stride, stride, 1], 'SAME')]

    return tf.concat(results, 3)


def _depthwise_conv(X, kernel=3, stride=1):

    in_channels = X.shape.as_list()[3]

    # kaiming uniform initialization
    maxval = math.sqrt(6.0/in_channels)

    W = tf.get_variable(
        'depthwise_kernel', [kernel, kernel, in_channels, 1],
        tf.float32, tf.random_uniform_initializer(-maxval, maxval)
    )

    return tf.nn.depthwise_conv2d(X, W, [1, stride, stride, 1], 'SAME')


def _shufflenet_unit(X, is_training, groups=3, stride=1):

    in_channels = X.shape.as_list()[3]
    result = X

    with tf.variable_scope('g_conv_1'):
        result = _group_conv(result, in_channels, groups)
        result = _batch_norm(result, is_training)
        result = _nonlinearity(result)

    with tf.variable_scope('channel_shuffle_2'):
        result = _channel_shuffle(result, groups)

    with tf.variable_scope('dw_conv_3'):
        result = _depthwise_conv(result, stride=stride)
        result = _batch_norm(result, is_training)

    with tf.variable_scope('g_conv_4'):
        result = _group_conv(result, in_channels, groups)
        result = _batch_norm(result, is_training)

    if stride < 2:
        result = tf.add(result, X)
    else:
        X = _avg_pooling(X)
        result = tf.concat([result, X], 3)

    result = _nonlinearity(result)
    return result


# first shufflenet unit is different from the rest
def _first_shufflenet_unit(X, is_training, groups, out_channels):

    in_channels = X.shape.as_list()[3]
    result = X
    out_channels -= in_channels

    with tf.variable_scope('g_conv_1'):
        result = _group_conv(result, out_channels, groups=1)
        result = _batch_norm(result, is_training)
        result = _nonlinearity(result)

    with tf.variable_scope('dw_conv_2'):
        result = _depthwise_conv(result, stride=2)
        result = _batch_norm(result, is_training)

    with tf.variable_scope('g_conv_3'):
        result = _group_conv(result, out_channels, groups)
        result = _batch_norm(result, is_training)

    X = _avg_pooling(X)
    result = tf.concat([result, X], 3)
    result = _nonlinearity(result)
    return result


def _affine(X, size):
    input_dim = X.shape.as_list()[1]

    # kaiming uniform initialization
    maxval = math.sqrt(6.0/input_dim)

    W = tf.get_variable(
        'kernel', [input_dim, size], tf.float32,
        tf.random_uniform_initializer(-maxval, maxval)
    )

    b = tf.get_variable(
        'bias', [size], tf.float32,
        tf.zeros_initializer()
    )

    return tf.nn.bias_add(tf.matmul(X, W), b)
