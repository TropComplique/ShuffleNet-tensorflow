import tensorflow as tf
from input_utils import _get_data
from parts_of_the_net import _mapping, _add_weight_decay


def get_squeezenet(optimizer, weight_decay=None, image_size=224, num_classes=1000):
    """Create a ShuffleNet computational graph.

    Arguments:
        optimizer: A Tensorflow optimizer.
        weight_decay: A scalar or None.
        image_size: An integer, input image size.
        num_classes: An integer.

    """

    graph = tf.Graph()
    with graph.as_default():

        with tf.variable_scope('control'):
            is_training = tf.placeholder_with_default(True, [], 'is_training')

        with tf.device('/cpu:0'), tf.variable_scope('input_pipeline'):
            data_init, x_batch, y_batch = _get_data(num_classes, image_size, is_training)

        with tf.variable_scope('inputs'):
            X = tf.placeholder_with_default(x_batch, [None, image_size, image_size, 3], 'X')
            Y = tf.placeholder_with_default(y_batch, [None, num_classes], 'Y')

        with tf.variable_scope('preprocessing'):
            f255 = tf.constant(255.0, tf.float32, [])
            mean = tf.constant([0.485, 0.456, 0.406], tf.float32, [3])
            std = tf.constant([0.229, 0.224, 0.225], tf.float32, [3])
            X /= f255
            X -= mean
            X /= std

        logits = _mapping(X, num_classes, is_training)

        with tf.variable_scope('softmax'):
            predictions = tf.nn.softmax(logits)

        with tf.variable_scope('log_loss'):
            log_loss = tf.losses.softmax_cross_entropy(Y, logits)

        if weight_decay is not None:
            with tf.variable_scope('weight_decay'):
                _add_weight_decay(weight_decay)

        with tf.variable_scope('total_loss'):
            total_loss = tf.losses.get_total_loss()

        with tf.variable_scope('optimizer'):
            grads_and_vars = optimizer.compute_gradients(total_loss)
            optimize = optimizer.apply_gradients(grads_and_vars)

        grad_summaries = tf.summary.merge(
            [tf.summary.histogram(v.name[:-2] + '_grad_hist', g)
             for g, v in grads_and_vars]
        )

        with tf.variable_scope('utilities'):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            assign_weights = _assign_weights()
            is_equal = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))

        summaries = _add_summaries()

    graph.finalize()
    ops = [
        data_init, predictions, log_loss, optimize,
        grad_summaries, init, saver, assign_weights,
        accuracy, summaries
    ]
    return graph, ops


def _add_summaries():
    # add histograms of all trainable variables

    summaries = []
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    for v in trainable_vars:
        summaries += [tf.summary.histogram(v.name[:-2] + '_hist', v)]

    return tf.summary.merge(summaries)


def _assign_weights():
    # add ops that can be used to load pretrained weights into the model

    assign_weights_dict = {}
    model_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)

    for v in model_vars:
        assign_weights_dict[v.name] = v.assign(
            tf.placeholder(tf.float32, v.shape, v.name[:-2])
        )

    return assign_weights_dict
