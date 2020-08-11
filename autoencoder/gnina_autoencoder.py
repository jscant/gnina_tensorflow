"""
Created on Tue Jun 23 14:45:32 2020

@author: scantleb
@brief: Main script for using an autoencoder to reduce the dimensionality of
gnina inputs.
"""

import argparse
import torch
from math import isnan
import molgrid
import numpy as np
import os
import tensorflow as tf
import time

from autoencoder import autoencoder_definitions
from matplotlib import pyplot as plt
from pathlib import Path
from autoencoder.calculate_embeddings import calculate_embeddings


class LoadConfig(argparse.Action):
    """Class for loading argparse arguments from a config file."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Overloaded function; See parent class."""
        
        if values is None:
            return
        config = Path(values).parents[1] / 'config'
        if not config.exists():
            raise RuntimeError(
                "No config file found in experiment's base directory ({})".format(
                    config))
        args = ''
        with open(config, 'r') as f:
            for line in f.readlines():
                chunks = line.split()
                if chunks[0] not in ['load_model',
                                     'absolute_save_path',
                                     'use_cpu',
                                     'binary_mask',
                                     'save_embeddings', ]:
                    args += '--{0} {1}\n'.format(*chunks)
                else:  # store_true args present a problem, loaded manually
                    if chunks[1] == 'True':
                        args += '--{0}\n'.format(chunks[0])
        print(args)
        parser.parse_args(args.split(), namespace)

        # args.load_model is always None if we do not do this, even when
        # it is specified using --load_model.
        namespace.load_model = values


def pickup(path):
    """Loads saved autoencoder.

    Arguments:
        path: location of saved weights and architecture

    Returns:
        AutoEncoderBase-derived object initialised with weights from saved
        checkpoint.
    """

    ae = tf.keras.models.load_model(
        path,
        custom_objects={
            'zero_mse': autoencoder_definitions.zero_mse,
            'nonzero_mse': autoencoder_definitions.nonzero_mse,
            'composite_mse': autoencoder_definitions.composite_mse,
            'nonzero_mae': autoencoder_definitions.nonzero_mae,
            'zero_mae': autoencoder_definitions.zero_mae,
            'approx_heaviside': autoencoder_definitions.approx_heaviside,
            'unbalanced_loss': autoencoder_definitions.unbalanced_loss,
        }
    )

    # Bug with add_loss puts empty dict at the end of model._layers which
    # interferes with some functionality (such as
    # tf.keras.utils.plot_model)
    ae._layers = [layer for layer in ae._layers if isinstance(
        layer, tf.keras.layers.Layer)]
    return ae


def parse_command_line_args():
    """Parse command line args and return as dict.

    Returns a dictionary containing all args, default or otherwise; if 'pickup'
    is specified, as many args as are contained in the config file for that
    (partially) trained model are loaded, otherwise defaults are given.
    Command line args override args found in the config of model found in
    'pickup' directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'load_model', type=str, action=LoadConfig, nargs='?',
        help="""Load saved keras model. If specified, this should be the 
        directory containing the saved assets of a saved autoencoder. If
        specified, the options are loaded from the config file saved when
        the original model was trained; any options specified here will
        override the original options.
        """)
    parser.add_argument("--data_root", '-r', type=str, required=False,
                        default='')
    parser.add_argument("--train", '-t', type=str, required=False)
    parser.add_argument('--encoding_size', '-e', type=int, required=False,
                        default=50)
    parser.add_argument(
        '--iterations', '-i', type=int, required=False)
    parser.add_argument(
        '--save_interval', type=int, required=False, default=10000)
    parser.add_argument(
        '--batch_size', '-b', type=int, required=False, default=16)
    parser.add_argument(
        '--model', '-m', type=str, required=False, default='single',
        help='Model architecture; one of single (SingleLayerAutoencoder' +
        '), dense (DenseAutoEncodcer) or auto (AutoEncoder)')
    parser.add_argument(
        '--optimiser', '-o', type=str, required=False, default='sgd')
    parser.add_argument(
        '--learning_rate', '-l', type=float, required=False)
    parser.add_argument(
        '--momentum', type=float, required=False, default=0.0)
    parser.add_argument(
        '--loss', type=str, required=False, default='mse')
    parser.add_argument(
        '--final_activation', type=str, required=False, default='sigmoid')
    parser.add_argument('--binary_mask', action='store_true')
    parser.add_argument(
        '--dimension', type=float, required=False, default=18.0),
    parser.add_argument(
        '--resolution', type=float, required=False, default=1.0),
    parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.')
    parser.add_argument(
        '--use_cpu', '-g', action='store_true')
    parser.add_argument(
        '--save_embeddings', action='store_true')

    

    args = parser.parse_args()

    autoencoder = None
    if args.load_model is not None:  # Load a model
        autoencoder = pickup(args.load_model)

    #args.train = Path(args.train).resolve()
    #args.data_root = Path(args.data_root).resolve()
    return autoencoder, args


def main():
    # Parse and sanitise command line args
    ae, args = parse_command_line_args()

    # For use later when defining model
    architectures = {'single': autoencoder_definitions.SingleLayerAutoEncoder,
                     'dense': autoencoder_definitions.DenseAutoEncoder,
                     'auto': autoencoder_definitions.AutoEncoder}

    molgrid.set_gpu_enabled(1-args.use_cpu)
    arg_str = '\n'.join(
        ['{0} {1}'.format(arg, getattr(args, arg)) for arg in vars(args)])

    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if isinstance(slurm_job_id, str):
        slurm_log_file = Path.home() / 'slurm_{}.out'.format(slurm_job_id)
        arg_str += '\nslurm_job_id {0}\nslurm_log_file {1}\n'.format(
            slurm_job_id, slurm_log_file)
        save_path = Path(args.save_path, slurm_job_id).resolve()
    else:
        save_path = Path(args.save_path, str(int(time.time()))).resolve()

    if args.momentum > 0 and args.optimiser.lower() not in ['sgd', 'rmsprop']:
        raise RuntimeError(
            'Momentum only used for RMSProp and SGD optimisers.')
    if not Path(args.train).exists():
        raise RuntimeError('{} does not exist.'.format(args.train))

    Path(save_path, 'checkpoints').mkdir(parents=True, exist_ok=True)

    arg_str += '\nabsolute_save_path {}\n'.format(save_path)
    print(arg_str)

    with open(save_path / 'config', 'w') as f:
        f.write(arg_str)

    tf.keras.backend.clear_session()

    # Setup libmolgrid to feed Examples into tensorflow objects
    e = molgrid.ExampleProvider(
        data_root=str(args.data_root), balanced=False, shuffle=True)
    e.populate(str(args.train))

    gmaker = molgrid.GridMaker(
        binary=args.binary_mask,
        dimension=args.dimension,
        resolution=args.resolution)

    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (args.batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    # Train autoencoder
    zero_losses, nonzero_losses, losses = [], [], []

    opt_args = {'lr': args.learning_rate, 'loss': args.loss}
    if args.momentum > 0:
        opt_args['momentum'] = args.momentum

    if ae is None: # No loaded model
        ae = architectures[args.model](
            dims,
            encoding_size=args.encoding_size,
            optimiser=args.optimiser,
            **opt_args)

    ae.summary()

    if not args.loss in ['composite_mse', 'unbalanced_loss']:
        tf.keras.utils.plot_model(
            ae, save_path / 'model.png', show_shapes=True)

    loss_ratio = 0.5
    loss_log = 'iteration {} nonzero_mae zero_mae nonzero_mean\n'.format(
        args.loss)
    print('Starting training cycle...')
    print('Working directory: {}'.format(save_path))

    for iteration in range(args.iterations):
        if not (iteration + 1) % args.save_interval \
                and iteration < args.iterations - 1:

            checkpoint_path = Path(
                save_path,
                'checkpoints',
                'ckpt_model_{}'.format(iteration + 1)
            )
            ae.save(checkpoint_path)

        batch = e.next_batch(args.batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=False)

        input_tensor_numpy = input_tensor.tonumpy()

        mean_nonzero = np.mean(
            input_tensor_numpy[np.where(input_tensor_numpy > 0)])

        x_inputs = {'input_image': input_tensor_numpy}
        if args.loss == 'composite_mse':
            x_inputs['frac'] = tf.constant(
                loss_ratio, shape=(args.batch_size,))

        loss = ae.train_on_batch(
            x_inputs,
            {'reconstruction': input_tensor_numpy},
            return_dict=True)

        zero_mae = loss['reconstruction_zero_mae']
        nonzero_mae = loss['reconstruction_nonzero_mae']
        if isnan(nonzero_mae):
            nonzero_mae = nonzero_losses[-1] if len(nonzero_losses) else 1.

        if zero_mae > 1e-5:
            loss_ratio = min(50, nonzero_mae / zero_mae)
        else:
            loss_ratio = 50

        if isnan(loss_ratio):
            loss_ratio = 0.5

        loss_str = '{0} {1:0.5f} {2:0.5f} {3:0.5f} {4:0.5f}'.format(
            iteration, loss['loss'], nonzero_mae, zero_mae, mean_nonzero)

        print(loss_str)
        loss_log += loss_str + '\n'
        if not iteration % 10:
            with open(save_path / 'loss_log.txt', 'w') as f:
                f.write(loss_log[:-1])

        zero_losses.append(zero_mae)
        nonzero_losses.append(nonzero_mae)
        losses.append(loss['loss'])
    print('Finished training.')

    # Save final trained autoencoder
    checkpoint_path = Path(
        save_path, 'checkpoints', 'final_model_{}'.format(args.iterations))
    ae.save(checkpoint_path)

    # Plot zero, nonzero mse
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Batches')
    ax2 = ax1.twinx()
    axes = [ax1, ax2]
    cols = ['r-', 'b-']
    labels = ['Zero_MAE', 'Nonzero_MAE']
    lines = []
    for idx, losses in enumerate([zero_losses, nonzero_losses]):
        gap = 100
        losses = [np.mean(losses[n:n+gap]) for n in range(0, len(losses), gap)]
        line, = axes[idx].plot(
            np.arange(len(losses))*gap, losses, cols[idx], label=labels[idx])
        axes[idx].set_ylabel('Loss')
        lines.append(line)
    ax1.legend(lines, [line.get_label() for line in lines])
    fig.savefig(save_path / 'zero_nonzero_losses.png')

    # Plot composite mse
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Batches')
    axes = [ax1]
    for idx, losses in enumerate([losses]):
        gap = 100
        losses = [np.mean(losses[n:n+gap]) for n in range(0, len(losses), gap)]
        axes[idx].plot(np.arange(len(losses))*gap, losses)
        axes[idx].set_ylabel('Loss')
    ax1.legend([args.loss])
    fig.savefig(save_path / 'composite_loss.png')

    if args.save_embeddings:
        # Save encodings in serialised format
        print('Saving encodings...')
        encodings_dir = save_path / 'encodings'
        encodings_dir.mkdir(exist_ok=True, parents=True)
        serialised_encodings = calculate_embeddings(
            ae, gmaker, input_tensor, args.data_root, args.train)
        for receptor_path, ligands in serialised_encodings.items():
            fname = receptor_path.split('/')[-1].split('.')[0] + '.bin'
            with open(encodings_dir / fname, 'wb') as f:
                f.write(ligands)


if __name__ == '__main__':
    main()
