import tensorflow as tf
import shutil
import os
import json
import time
import sys
from tqdm import tqdm
import argparse
from shufflenet import get_shufflenet


parser = argparse.ArgumentParser()
parser.add_argument(
    '--run', type=int, default=0,
    help=('An integer that determines a folder where logs and '
          'the fitted model will be saved.')
)
parser.add_argument(
    '--reset', action='store_true',
    help=('Use this if you want to ignore the previously saved model. '
          '(Warning: the saved model and all logs will be destroyed.)')
)
parser.add_argument(
    '--train_tfrecords', type=str, default='/home/ubuntu/data/tiny-imagenet-200/train.tfrecords',
    help='A string, path to tfrecords train dataset file.'
)
parser.add_argument(
    '--val_tfrecords', type=str, default='/home/ubuntu/data/tiny-imagenet-200/val.tfrecords',
    help='A string, path to tfrecords validation dataset file.'
)
parser.add_argument(
    '--num_epochs', type=int, default=35,
    help='An integer.'
)
parser.add_argument(
    '--batch_size', type=int, default=200,
    help='An integer.'
)
parser.add_argument(
    '--steps_per_epoch', type=int, default=500,
    help='An integer, number of optimization steps per epoch.'
)
parser.add_argument(
    '--validation_steps', type=int, default=50,
    help='An integer, number of batches from validation dataset to evaluate on.'
)
parser.add_argument(
    '--lr_patience', type=int, default=4,
    help='An integer, patience for "reduce on plateau" learning rate scheduler.'
)
parser.add_argument(
    '--lr_threshold', type=float, default=0.01,
    help='An float number, threshold for learning rate scheduler.'
)
parser.add_argument(
    '--patience', type=int, default=10,
    help=('An integer, number of epochs before early stopping '
          'if test accuracy isn\'t improving.')
)
parser.add_argument(
    '--threshold', type=float, default=0.01,
    help='An float number, threshold for early stopping.'
)
parser.add_argument(
    '--initial_lr', type=float, default=1e-1,
    help='A floar number, initial learning rate.'
)
parser.add_argument(
    '--weight_decay', type=float, default=4e-3,
    help='A floar number.'
)
parser.add_argument(
    '--groups', type=int, default=3,
    help=('An integer, number of groups in group convolutions, '
          'only possible values are: 1, 2, 3, 4, 8.')
)
parser.add_argument(
    '--dropout', type=float, default=0.5,
    help='A floar number, dropout rate before last linear layer.'
)
parser.add_argument(
    '--complexity_scale_factor', type=float, default=0.75,
    help=('It determines the width of the network, '
          'only possible values are: 0.25, 0.5, 0.75, 1.0.')
)
FLAGS = parser.parse_args()


def train():

    # folders for logging and saving
    dir_to_log = os.path.join('logs', 'run' + str(FLAGS.run))
    dir_to_save = os.path.join('saved', 'run' + str(FLAGS.run))

    print('\nTraining logs and summaries will be in', dir_to_log)
    print('Saved model will be in', dir_to_save, '\n')

    # create these folders
    if FLAGS.reset and os.path.exists(dir_to_log):
        shutil.rmtree(dir_to_log)
    if FLAGS.reset and os.path.exists(dir_to_save):
        shutil.rmtree(dir_to_save)
    if not os.path.exists(dir_to_log):
        os.makedirs(dir_to_log)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    # files with losses and config
    training_info_file = os.path.join(dir_to_log, 'training_info.txt')
    model_config_file = os.path.join(dir_to_log, 'model_config.txt')
    print('Training/validation evaluations will be in', training_info_file)
    print('Model config will be in', model_config_file, '\n')

    # create the graph and start a session
    graph, ops = get_shufflenet(
        FLAGS.initial_lr, FLAGS.weight_decay,
        FLAGS.groups, FLAGS.dropout,
        FLAGS.complexity_scale_factor
    )
    sess = tf.Session(graph=graph)
    writer = tf.summary.FileWriter(dir_to_log, sess.graph)
    print('\nCreated the graph and started a session!')

    # check if to continue training
    warm = os.path.exists(training_info_file)
    if warm and not FLAGS.reset:
        print('Restoring previously saved model and continuing training.\n')
        initial_epoch = sum(1 for line in open(training_info_file))
        try:
            ops['saver'].restore(sess, os.path.join(dir_to_save, 'model'))
        except:
            print('\nCan\'t restore the saved model, '
                  'maybe architectures don\'t match.')
            sys.exit()
    else:
        print('Training model from scratch.\n')
        initial_epoch = 1
        sess.run(ops['init_variables'])

    # initialize data sources
    data_dict = {
        'input_pipeline/train_file:0': FLAGS.train_tfrecords,
        'input_pipeline/val_file:0': FLAGS.val_tfrecords,
        'input_pipeline/batch_size:0': FLAGS.batch_size
    }
    sess.run(ops['init_data'], data_dict)

    losses = []
    training_epochs = range(
        initial_epoch,
        initial_epoch + FLAGS.num_epochs
    )

    # begin training
    try:
        for epoch in training_epochs:

            start_time = time.time()
            running_loss, running_accuracy = 0.0, 0.0
            sess.run(ops['train_init'])

            # at zeroth step also collect metadata and summaries
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE
            )
            run_metadata = tf.RunMetadata()

            # do epoch's zeroth step
            _, batch_loss, batch_accuracy, summary, grad_summary = sess.run([
                ops['optimize'], ops['log_loss'], ops['accuracy'],
                ops['summaries'], ops['grad_summaries']
            ], options=run_options, run_metadata=run_metadata)
            running_loss += batch_loss
            running_accuracy += batch_accuracy

            print('epoch', epoch)
            training_steps = tqdm(
                range(1, FLAGS.steps_per_epoch),
                initial=1, total=FLAGS.steps_per_epoch
            )

            # main training loop
            for step in training_steps:

                _, batch_loss, batch_accuracy = sess.run([
                    ops['optimize'], ops['log_loss'], ops['accuracy']
                ])
                running_loss += batch_loss
                running_accuracy += batch_accuracy

            # evaluate on the validation set
            val_loss, val_accuracy = _evaluate(
                sess, ops, FLAGS.validation_steps
            )
            train_loss = running_loss/FLAGS.steps_per_epoch
            train_accuracy = running_accuracy/FLAGS.steps_per_epoch

            # collect all losses and accuracies
            losses += [(
                epoch, train_loss, val_loss,
                train_accuracy, val_accuracy, time.time() - start_time
            )]
            writer.add_run_metadata(run_metadata, str(epoch))
            writer.add_summary(summary, epoch)
            writer.add_summary(grad_summary, epoch)

            print('loss: {0:.3f}, val_loss: {1:.3f}, '
                  'acc: {2:.3f}, val_acc: {3:.3f}, time: {4:.3f}\n'.format(*losses[-1][1:]))

            # consider a possibility of early stopping
            if _is_early_stopping(losses, FLAGS.patience, FLAGS.threshold):
                print('Early stopping!')
                break

            # consider a possibility of reducing learning rate by some factor
            _reduce_lr_on_plateau(
                sess, ops, losses,
                FLAGS.lr_patience, FLAGS.lr_threshold
            )
    except (KeyboardInterrupt, SystemExit):
        # you can interrupt training by ctrl-c,
        # your model will be saved
        print(' Interruption detected, exiting the program...')

    print('Writing logs and saving the trained model.')
    _write_training_info(
        FLAGS, losses, warm,
        training_info_file, model_config_file
    )
    ops['saver'].save(sess, os.path.join(dir_to_save, 'model'))
    sess.close()


def _evaluate(sess, ops, validation_steps):

    val_loss, val_accuracy = 0.0, 0.0
    sess.run(ops['val_init'])

    for i in range(validation_steps):
        batch_loss, batch_accuracy = sess.run(
            [ops['log_loss'], ops['accuracy']],
            {'control/is_training:0': False}
        )
        val_loss += batch_loss
        val_accuracy += batch_accuracy

    val_loss /= validation_steps
    val_accuracy /= validation_steps
    return val_loss, val_accuracy


# it decides if training must stop
def _is_early_stopping(losses, patience=10, threshold=0.01):

    # get validation set accuracies
    accuracies = [x[4] for x in losses]

    if len(losses) > (patience + 4):
        # running average
        average = (accuracies[-(patience + 4)] +
                   accuracies[-(patience + 3)] +
                   accuracies[-(patience + 2)] +
                   accuracies[-(patience + 1)] +
                   accuracies[-patience])/5.0
        return accuracies[-1] < average + threshold
    else:
        return False


def _reduce_lr_on_plateau(
        sess, ops, losses,
        patience=10, threshold=0.01
        ):

    # get validation set accuracies
    accuracies = [x[4] for x in losses]

    if len(losses) > (patience + 4):
        # running average
        average = (accuracies[-(patience + 4)] +
                   accuracies[-(patience + 3)] +
                   accuracies[-(patience + 2)] +
                   accuracies[-(patience + 1)] +
                   accuracies[-patience])/5.0
        if accuracies[-1] < (average + threshold):
            sess.run(ops['drop_learning_rate'])
            print('Learning rate is dropped!\n')


def _write_training_info(
        FLAGS, losses, warm,
        training_info_file, model_config_file
        ):

    mode = 'a' if warm else 'w'
    with open(training_info_file, mode) as f:

        # if file is new then add columns
        if not warm:
            columns = ('epoch,train_loss,val_loss,'
                       'train_accuracy,val_accuracy,time\n')
            f.write(columns)

        for i in losses:
            values = ('{0},{1:.3f},{2:.3f},'
                      '{3:.3f},{4:.3f},{5:.3f}\n').format(*i)
            f.write(values)

    with open(model_config_file, mode) as f:
        FLAGS_dict = vars(FLAGS)
        # len(losses) equals to the number of passed full epochs
        FLAGS_dict['num_epochs'] = len(losses)
        json.dump(FLAGS_dict, f)
        f.write('\n')


if __name__ == '__main__':
    train()
