"""
Created on Sat Jun 20 14:38:14 2020

@author: scantleb
@brief: Function to train autoencoders defined in autoencoder_definitions.
"""

import time
from math import isnan
from pathlib import Path

import calculate_distances as cd
import molgrid
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from utilities.gnina_functions import format_time, wipe_directory, \
    print_with_overwrite


def train(model, data_root, train_types, iterations, batch_size,
          dimension, resolution, loss_fn, save_path=None,
          overwrite_checkpoints=False, ligmap=None, recmap=None,
          save_interval=-1, binary_mask=False, silent=False,
          loss_log=None, starting_iter=0):
    """Train an autoencoder.
    
    Arguments:
        model: compiled autoencoder for training
        data_root: path to which all paths in train_types are relative
        train_types: text files with lines of the format:
            <label> <path_to_receptor> <path_to_ligand>
        iterations: number of batches to learn from during training
        batch_size: number of training examples in each batch
        save_path: where to save results and checkpoints
        dimension: length of cube in which ligand is situated in Angstroms
        resolution: resolution of input cube in Angstroms
        loss_fn: (string) name of keras loss fn or 'composite_mse'
        ligmap: location of text file containing ligand smina types grouped by
            line to be used in input grid construction
        recmap: location of text file containing receptor smina types grouped by
            line to be used in input grid construction
        save_interval: interval (in batches) on which model weights are saved as
            a checkpoint
        overwrite_checkpoints: saved model states overwrite previously saved
            ones from earlier in training
        binary_mask: instead of real numbers, input grid is binary where a 1
            indicates that the real input would have had a value greater than
            zero
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
    example_provider_kwargs = {
        'data_root': str(Path(data_root).expanduser()), 'balanced': False,
        'shuffle': True, 'cache_structs': False
    }
    if ligmap is None or recmap is None:
        # noinspection PyArgumentList
        e = molgrid.ExampleProvider(
            **example_provider_kwargs
        )
        rec_channels = 14
    else:
        rec_typer = molgrid.FileMappedGninaTyper(recmap)
        lig_typer = molgrid.FileMappedGninaTyper(ligmap)
        e = molgrid.ExampleProvider(
            rec_typer, lig_typer, **example_provider_kwargs)
        with open(recmap, 'r') as f:
            rec_channels = len([line for line in f.readlines() if len(line)])
    e.populate(str(Path(train_types).expanduser()))

    # noinspection PyArgumentList
    gmaker = molgrid.GridMaker(
        binary=binary_mask,
        dimension=dimension,
        resolution=resolution)

    # noinspection PyArgumentList
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    # Train autoencoder
    zero_losses, nonzero_losses, losses = [], [], []

    # Composite mse loss ratio
    loss_ratio = 0.5

    # Are we loading previous loss history or starting afresh?
    if loss_log is None:
        loss_log = 'iteration loss nonzero_mae zero_mae nonzero_mean learning_rate\n'

    if not silent and save_path is not None:
        save_path = Path(save_path)
        print('Working directory: {}'.format(save_path))

    previous_checkpoint = None
    start_time = time.time()
    for iteration in range(starting_iter, iterations):
        if save_path is not None and not (iteration + 1) % save_interval \
                and iteration < iterations - 1:
            checkpoint_path = Path(
                save_path,
                'checkpoints',
                'ckpt_model_{}'.format(iteration + 1)
            )
            model.save(checkpoint_path)

            if overwrite_checkpoints:
                if previous_checkpoint is not None:
                    wipe_directory(previous_checkpoint)
                previous_checkpoint = checkpoint_path

        # Use learning rate scheduler to find learning rate
        if isinstance(model.learning_rate_schedule,
                      tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = model.learning_rate_schedule(iteration)
            K.set_value(model.optimizer.learning_rate, lr)

        batch = e.next_batch(batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=False)

        input_tensor_numpy = np.minimum(input_tensor.tonumpy(), 1.0)

        x_inputs = {'input_image': input_tensor_numpy}

        if loss_fn == 'composite_mse':
            x_inputs['frac'] = tf.constant(
                loss_ratio, shape=(batch_size,))

        if loss_fn == 'distance_mse':
            spatial_information = np.zeros(
                (batch_size, *dims[1:]), dtype='float32')
            for i in range(batch_size):
                fortran_tensor = np.asfortranarray(
                    input_tensor_numpy[i, :, :, :, :])
                spatial_information[i, :, :, :] = cd.calculate_distances(
                    rec_channels, fortran_tensor, resolution)
            distances = np.stack([spatial_information] * dims[0], axis=1)
            x_inputs['distances'] = distances

        mean_nonzero = np.mean(
            input_tensor_numpy[np.where(input_tensor_numpy > 0)])

        loss = model.train_on_batch(
            x_inputs,
            {'reconstruction': input_tensor_numpy},
            return_dict=True)

        zero_mae = loss['reconstruction_trimmed_zero_mae']
        nonzero_mae = loss['reconstruction_trimmed_nonzero_mae']
        if isnan(nonzero_mae):
            nonzero_mae = nonzero_losses[-1] if len(nonzero_losses) else 1.

        if zero_mae > 1e-5:
            loss_ratio = min(50, nonzero_mae / zero_mae)
        else:
            loss_ratio = 50

        if isnan(loss_ratio):
            loss_ratio = 0.5

        lr = K.get_value(model.optimizer.learning_rate)

        loss_str = '{0} {1:0.5f} {2:0.5f} {3:0.5f} {4:0.5f} {5:0.8f}'.format(
            iteration, loss['loss'], nonzero_mae, zero_mae, mean_nonzero, lr)

        time_elapsed = time.time() - start_time
        time_per_iter = time_elapsed / (iteration + 1 - starting_iter)
        time_remaining = time_per_iter * (iterations - iteration - 1)
        formatted_eta = format_time(time_remaining)

        if not iteration and not silent:
            print('\n')

        if not silent:
            console_output = ('Iteration: {0}/{1} | Time elapsed {6} | '
                              'Time remaining: {7}'
                              '\nLoss ({2}): {3:0.4f} | Non-zero MAE: {4:0.4f} '
                              '| Zero MAE: {5:0.4f} | Learning rate: {8:.3e}')
            console_output = console_output.format(
                iteration, iterations, loss_fn, loss['loss'], nonzero_mae,
                zero_mae, format_time(time_elapsed), formatted_eta, lr
            )
            print_with_overwrite(console_output)

        if save_path is not None:
            loss_log += loss_str + '\n'
            if not iteration % 10:
                with open(save_path / 'loss_log.txt', 'w') as f:
                    f.write(loss_log[:-1])

        zero_losses.append(zero_mae)
        nonzero_losses.append(nonzero_mae)
        losses.append(loss['loss'])

    if save_path is not None:
        # Save final trained autoencoder
        checkpoint_path = Path(
            save_path, 'checkpoints', 'final_model_{}'.format(iterations))
        model.save(checkpoint_path)

        if overwrite_checkpoints and previous_checkpoint is not None:
            wipe_directory(previous_checkpoint)

    return losses, zero_losses, nonzero_losses
