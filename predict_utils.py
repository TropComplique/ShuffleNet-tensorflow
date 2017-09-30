import tensorflow as tf
import os


def predict_proba(graph, ops, X, run):
    """Predict probabilities with a fitted model.

    Arguments:
        graph: A Tensorflow graph.
        ops: A dict of ops of the graph.
        X: A numpy array of shape [n_samples, image_size, image_size, 3]
            and of type 'float32'.
        run: An integer that determines a folder where a fitted model
            is saved.

    Returns:
        predictions: A numpy array of shape [n_samples, n_classes]
            and of type 'float32'.
    """
    sess = tf.Session(graph=graph)
    ops['saver'].restore(sess, os.path.join('saved', 'run' + str(run) + '/model'))
    
    feed_dict = {'inputs/X:0': X, 'control/is_training:0': False}
    predictions = sess.run(ops['predictions'], feed_dict)
    
    sess.close()
    return predictions
