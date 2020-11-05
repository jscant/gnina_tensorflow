"""
Created on Sat Jun 20 14:38:14 2020

@author: scantleb
@brief: Function to train autoencoders defined in autoencoder_definitions.
"""

import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import backend as K

from utilities.gnina_functions import format_time, wipe_directory, \
    print_with_overwrite


def train(model, iterations, loss_fn, save_path=None,
          overwrite_checkpoints=False, save_interval=-1,
          silent=False, loss_log=None, starting_iter=0):
    """Train an autoencoder.

    Arguments:
        model: compiled autoencoder for training
        iterations: number of batches to learn from during training
        save_path: where to save results and checkpoints
        loss_fn: (string) name of keras loss fn or 'composite_mse'
        save_interval: interval (in batches) on which model weights are saved as
            a checkpoint
        overwrite_checkpoints: saved model states overwrite previously saved
            ones from earlier in training
        silent: when True, print statements are suppressed, no output is written
            to disk (checkpoints, loss history)
        loss_log: string containing losses up to the point of the start of
            training; if None, a new string (and loss_log.txt file) is
            generated.
        starting_iter: which iteration to start training from; used for
            resumption of training from a saved checkpoint

    Returns:
        Three lists containing the loss history, mean average error for inputs
        equal to zero and mean average error for inputs greater than zero,
        respectively.
    """
    # Setup libmolgrid to feed Examples into tensorflow objects
    save_interval = save_interval if save_interval > 0 else iterations + 10

    # Train autoencoder
    zero_losses, nonzero_losses, losses = [], [], []

    # Are we loading previous loss history or starting afresh?
    if loss_log is None:
        loss_log = 'iteration loss nonzero_mae zero_mae nonzero_mean ' \
                   'learning_rate\n'

    if not silent and save_path is not None:
        save_path = Path(save_path)
        print('Working directory: {}'.format(save_path))

    previous_checkpoint = None
    ratio = 1.0
    time_elapsed, time_per_iter, time_remaining = tuple(['--'] * 3)
    for iteration in range(starting_iter, iterations):
        if save_path is not None and not (iteration + 1) % save_interval \
                and iteration < iterations - 1:
            checkpoint_path = Path(
                save_path,
                'checkpoints',
                'ckpt_model_{}'.format(iteration + 1),
                'ckpt_model_{}'.format(iteration + 1)
            )
            model.save_weights(checkpoint_path, save_format='tf')

            if overwrite_checkpoints:
                if previous_checkpoint is not None:
                    wipe_directory(previous_checkpoint)
                previous_checkpoint = checkpoint_path

        # Use learning rate scheduler to find learning rate
        if isinstance(model.learning_rate_schedule,
                      tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = model.learning_rate_schedule(iteration)
            K.set_value(model.opt.learning_rate, lr)
        else:
            lr = K.get_value(model.opt.learning_rate)

        if not iteration and not silent:
            print('\n')

        if (iteration - starting_iter) > 5:
            time_elapsed = time.time() - start_time
            time_per_iter = time_elapsed / (iteration + 1 - starting_iter - 4)
            time_remaining = time_per_iter * (iterations - iteration - 1)
            time_remaining = format_time(time_remaining)
            time_elapsed = format_time(time_elapsed)
        else:
            start_time = time.time()

        loss, metrics = model.train_step(ratio=ratio)
        loss = float(tf.reduce_mean(loss))

        zero_mae = metrics.get('trimmed_zero_mae')
        nonzero_mae = metrics.get('trimmed_nonzero_mae')

        ratio = nonzero_mae / zero_mae

        loss_str = '{0} {1:0.5f} {2:0.5f} {3:0.5f} {4:.3f}'.format(
            iteration, loss, nonzero_mae, zero_mae, lr)

        if not silent:
            console_output = ('Iteration: {0}/{1} | Time elapsed {6} | '
                              'Time remaining: {7} | Learning rate: {8:.3e}'
                              '\nLoss ({2}): {3:0.4f} | Non-zero MAE: {4:0.4f} '
                              '| Zero MAE: {5:0.4f}\n').format(
                iteration, iterations, loss_fn, loss, nonzero_mae, zero_mae,
                time_elapsed, time_remaining, lr)
            if iteration == starting_iter:
                print()
            print_with_overwrite(console_output)

        if save_path is not None:
            loss_log += loss_str + '\n'
            if not iteration % 10:
                with open(save_path / 'loss_log.txt', 'w') as f:
                    f.write(loss_log[:-1])

        zero_losses.append(zero_mae)
        nonzero_losses.append(nonzero_mae)
        losses.append(loss)

    if save_path is not None:
        # Save final trained autoencoder
        checkpoint_path = Path(
            save_path, 'checkpoints', 'final_model_{}'.format(iterations))
        model.save_weights(checkpoint_path, save_format='h5')

        if overwrite_checkpoints and previous_checkpoint is not None:
            wipe_directory(previous_checkpoint)

    return losses, zero_losses, nonzero_losses
