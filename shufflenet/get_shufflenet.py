import tensorflow as tf
from .input_pipeline import _get_data
from .architecture import _mapping


MOMENTUM = 0.9
USE_NESTEROV = True
LR_REDUCE_FACTOR = 0.1

# tiny-imagenet dataset has 200 categories
# and all images have 64x64 size,
# but i use 56x56 images because i do
# data augmentation by making random crops
IMAGE_SIZE = 56
NUM_CLASSES = 200


def get_shufflenet(
        initial_lr=1e-2, weight_decay=1e-4,
        groups=3, dropout=0.5, complexity_scale_factor=0.75
        ):
    """Create a ShuffleNet computational graph.

    Arguments:
        initial_lr: A floar number.
        weight_decay: A floar number.
        groups: An integer, number of groups in group convolutions,
            only possible values are: 1, 2, 3, 4, 8.
        dropout: A floar number, dropout rate before last linear layer.
        complexity_scale_factor: A floar number, to customize the network
            to a desired complexity you can apply a scale factor,
            in the original paper they are considering
            scale factor values: 0.25, 0.5, 1.0.
            It determines the width of the network.

    Returns:
        graph: A Tensorflow graph.
        ops: A dict with ops.
    """

    graph = tf.Graph()
    with graph.as_default():

        with tf.variable_scope('control'):
            is_training = tf.placeholder_with_default(True, [], 'is_training')

        with tf.device('/cpu:0'), tf.variable_scope('input_pipeline'):
            data = _get_data(NUM_CLASSES, IMAGE_SIZE)

        with tf.variable_scope('inputs'):
            X = tf.placeholder_with_default(
                data['x_batch'], [None, IMAGE_SIZE, IMAGE_SIZE, 3], 'X'
            )
            Y = tf.placeholder_with_default(
                data['y_batch'], [None, NUM_CLASSES], 'Y'
            )

        with tf.variable_scope('preprocessing'):
            mean = tf.constant([0.485, 0.456, 0.406], tf.float32, [3])
            std = tf.constant([0.229, 0.224, 0.225], tf.float32, [3])
            # these values are taken from here:
            # http://pytorch.org/docs/master/torchvision/models.html,
            # but they are not very important, i think.
            X -= mean
            X /= std

        logits = _mapping(
            X, is_training, NUM_CLASSES,
            groups, dropout, complexity_scale_factor
        )

        with tf.variable_scope('softmax'):
            predictions = tf.nn.softmax(logits)

        with tf.variable_scope('log_loss'):
            log_loss = tf.losses.softmax_cross_entropy(Y, logits)

        with tf.variable_scope('weight_decay'):
            _add_weight_decay(weight_decay)

        with tf.variable_scope('total_loss'):
            total_loss = tf.losses.get_total_loss()

        with tf.variable_scope('learning_rate'):
            learning_rate = tf.Variable(
                initial_lr, trainable=False,
                dtype=tf.float32, name='lr'
            )
            drop_learning_rate = tf.assign(
                learning_rate, LR_REDUCE_FACTOR*learning_rate
            )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate, momentum=MOMENTUM, use_nesterov=USE_NESTEROV
            )
            grads_and_vars = optimizer.compute_gradients(total_loss)
            optimize = optimizer.apply_gradients(grads_and_vars)

        grad_summaries = tf.summary.merge(
            [tf.summary.histogram(v.name[:-2] + '_grad_hist', g)
             for g, v in grads_and_vars]
        )

        with tf.variable_scope('utilities'):
            init_variables = tf.global_variables_initializer()
            saver = tf.train.Saver()
            is_equal = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))

        summaries = _add_summaries()

    graph.finalize()
    ops = {
        # initialization
        'init_variables': init_variables,
        'init_data': data['init_data'],
        'train_init': data['train_init'],
        'val_init': data['val_init'],

        # training
        'optimize': optimize, 'drop_learning_rate': drop_learning_rate,

        # evaluation
        'predictions': predictions,
        'log_loss': log_loss, 'accuracy': accuracy,
        'summaries': summaries, 'grad_summaries': grad_summaries,
        'saver': saver
    }
    return graph, ops


def _add_summaries():
    # add histograms of all trainable variables

    summaries = []
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    for v in trainable_vars:
        summaries += [tf.summary.histogram(v.name[:-2] + '_hist', v)]

    return tf.summary.merge(summaries)


def _add_weight_decay(weight_decay):
    # add L2 regularization to all trainable kernel weights

    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )

    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = [v for v in trainable if 'kernel' in v.name]

    for K in kernels:
        l2_loss = tf.multiply(
            weight_decay, tf.nn.l2_loss(K)
        )
        tf.losses.add_loss(l2_loss)
