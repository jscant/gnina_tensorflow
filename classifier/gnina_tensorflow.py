# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:46:06 2020

@author: scantleb
@brief: Main script for training and inference with gnina-based neural
networks (https://github.com/gnina/gnina) .

Requirements: libmolgrid, pytorch (1.3.1), tensorflow 2.2.0+
"""

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import molgrid
import numpy as np
from tensorflow.keras.utils import plot_model

from autoencoder.parse_command_line_args import pickup, LoadConfigTrain
from classifier.inference import inference
from classifier.model_definitions import define_baseline_model, \
    define_densefs_model
from utilities.gnina_functions import process_batch, print_with_overwrite, \
    format_time


def main():
    # Create and parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'load_model', type=str, action=LoadConfigTrain, nargs='?',
        help='Load saved keras model. If specified, this should be the '
             'directory containing the assets of a saved classifier. '
             'If specified, the options are loaded from the config file '
             'saved when the original model was trained; any options '
             'specified in the command line will override the options '
             'loaded from the config file.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--data_root', '-r', type=str, required=False, default=Path.home(),
        help=('Path relative to which all paths in specified types files will '
              'be taken')
    )
    parser.add_argument(
        '--train', type=str, required=False,
        help=('Types file containing training examples including label, '
              'receptor path and ligand path')
    )
    parser.add_argument(
        '--test', type=str, required=False, help=(
            'Types file containing training examples including label, receptor'
            'path and ligand path')
    )
    parser.add_argument(
        '--densefs', '-d', action='store_true',
        help='Use DenseFS rather than Gnina (baseline)')
    parser.add_argument(
        '--iterations', '-i', type=int, required=False, default=25000,
        help='Number of batches to train on')
    parser.add_argument(
        '--batch_size', '-b', type=int, required=False, default=16,
        help='Number of training examples in each batch')
    parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.',
        help=('Directory where named folder will be created, in which '
              'results, models and config will be saved')
    )
    parser.add_argument(
        '--save_interval', type=int, default=10000,
        help='How often to save a snapshot of the model during training')
    parser.add_argument(
        '--use_cpu', '-g', action='store_true', help='Use CPU (not GPU)')
    parser.add_argument(
        '--use_densenet_bc', action='store_true',
        help='Use updated definition of DenseNet blocks (DenseNet-BC)')
    parser.add_argument(
        '--inference_on_training_set', action='store_true',
        help='Perform inference on training set')
    parser.add_argument(
        '--autoencoder', type=str, required=False,
        help=('Use trained autoencoder reconstruction as inputs for training '
              'and testing')
    )
    parser.add_argument(
        '--dimension', type=float, required=False, default=23.0,
        help=('Size of cube containing receptor atoms centred on ligand in '
              'Angstroms (default=23.0)')
    )
    parser.add_argument(
        '--resolution', type=float, required=False, default=0.5,
        help='Length of the side of each voxel in Angstroms (default=0.5)')
    parser.add_argument(
        '--binary_mask', action='store_true',
        help='Inputs are converted from real to binary (0 for 0, 1 for > 0)'
    )
    parser.add_argument(
        '--ligmap', type=str, required=False,
        help=('Text file containing space-delimited line with atom categories '
              'for each ligand channel input')
    )
    parser.add_argument(
        '--recmap', type=str, required=False,
        help=('Text file containing space-delimited line with atom categories '
              'for each receptor channel input'))
    parser.add_argument(
        '--name', type=str, required=False,
        help=('Name of folder to store results (default is current UTC time '
              'in seconds)')
    )
    parser.add_argument(
        '--seed', type=int, required=False,
        default=np.random.randint(0, int(1e7)),
        help=('Number used to seed molgrid; default is a random integer on the '
              'interval [0, 1e7)'))
    args = parser.parse_args()

    for item in vars(args):
        print(item, getattr(args, item))

    # We need to train or test
    if not (args.train or args.test) and not args.resume:
        raise RuntimeError('Please specify at least one of --train or '
                           '--test')
    else:
        # Check if types files exist
        for fname in [types for types in [args.train, args.test]
                      if types is not None]:
            if not Path(fname).exists():
                raise RuntimeError('{} does not exist.'.format(fname))

    data_root = Path(args.data_root).resolve()
    train_types = Path(args.train).resolve()
    test_types = Path(args.test).resolve() if args.test is not None else None

    if args.name is None:
        folder = os.getenv('SLURM_JOB_ID')
        if not isinstance(folder, str):
            folder = str(int(time.time()))
    else:
        folder = args.name

    if args.load_model is None:
        args.save_path = Path(args.save_path, folder).expanduser().resolve()
    else:
        args.save_path = Path(args.save_path).expanduser().resolve()

    # Use cpu rather than gpu if specified
    molgrid.set_gpu_enabled(1 - args.use_cpu)
    molgrid.set_random_seed(args.seed)

    # If specified, use autoencoder reconstructions to train/test
    autoencoder = None
    if isinstance(args.autoencoder, str):
        if Path(args.autoencoder).is_dir():
            autoencoder = pickup(args.autoencoder)

    gap = 100  # Window to average training loss over (in batches)

    # Setup libmolgrid to feed Examples into tensorflow objects
    example_provider_kwargs = {
        'data_root': str(Path(data_root).expanduser()), 'balanced': True,
        'shuffle': True, 'cache_structs': False
    }
    if args.ligmap is None:
        e = molgrid.ExampleProvider(
            **example_provider_kwargs)
    else:
        rec_typer = molgrid.FileMappedGninaTyper(args.recmap)
        lig_typer = molgrid.FileMappedGninaTyper(args.ligmap)
        e = molgrid.ExampleProvider(
            rec_typer, lig_typer, **example_provider_kwargs)

    e.populate(str(train_types))

    gmaker = molgrid.GridMaker(dimension=args.dimension,
                               resolution=args.resolution,
                               binary=args.binary_mask)
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (args.batch_size,) + dims

    labels = molgrid.MGrid1f(args.batch_size)
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    checkpoints_dir = args.save_path / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    barred_args = ['resume']
    losses_string = ''
    starting_iter = 0
    if args.resume:
        if not args.load_model:
            raise RuntimeError(
                '--resume must be used in conjunction with load_model')
        log_fname = Path(
            args.load_model).expanduser().parents[1] / 'loss_log.txt'
        starting_iter = int(str(Path(args.load_model).name).split('_')[-1])
        with open(log_fname, 'r') as f:
            losses_string = '\n'.join(
                f.read().split('\n')[:starting_iter + 1]) + '\n'
        barred_args.append('load_model')

    # We are ready to define our model and train
    losses = []

    if args.load_model is not None:  # Load a model
        model = pickup(args.load_model)
    else:
        if args.densefs:
            model = define_densefs_model(dims, bc=args.use_densenet_bc)
        else:
            model = define_baseline_model(dims)

    arg_str = '\n'.join(
        ['{0} {1}'.format(arg, getattr(args, arg)) for arg in vars(args)])
    model_str = ['Baseline', 'DenseFS'][args.densefs]
    plot_model(model, args.save_path / 'model.png', show_shapes=True)
    model.summary()
    with open(args.save_path / 'config', 'w') as f:
        f.write(arg_str)
    print(arg_str)

    # Logging process ID is useful for memory profiling (see utilities)
    gnina_tf_root = Path(__file__).expanduser().resolve().parents[1]
    with open(gnina_tf_root / 'process_ids.log', 'a') as f:
        f.write('{0} {1}\n'.format(os.getpid(), args.save_path))

    loss_history_fname = Path(args.save_path, 'loss_log.txt')
    start_time = time.time()

    for iteration in range(starting_iter, args.iterations):
        if (not (iteration + 1) % args.save_interval and
                iteration < args.iterations - 1):
            checkpoint_path = Path(
                args.save_path,
                'checkpoints',
                'ckpt_model_{}'.format(iteration + 1))

            model.save(checkpoint_path)

        # Data: e > gmaker > input_tensor > network (forward and backward pass)
        loss = process_batch(model, e, gmaker, input_tensor, labels,
                             train=True, autoencoder=autoencoder)

        # Save losses to disk
        if not isinstance(loss, float):
            loss = loss[0]
        losses.append(loss)
        losses_string += '{1} loss: {0:0.3f}\n'.format(loss, iteration)
        with open(loss_history_fname, 'w') as f:
            f.write(losses_string)

        time_elapsed = time.time() - start_time
        # Let's not div0
        time_per_iter = time_elapsed / (iteration - starting_iter + 1)
        time_remaining = time_per_iter * (args.iterations - iteration - 1)
        formatted_eta = format_time(time_remaining)

        if not iteration:
            print('\n')

        console_output = (
            'Iteration: {0}/{1} | loss: {2:0.4f}\nTime elapsed: {3} | Time remaining: {4}').format(
            iteration, args.iterations, loss, format_time(time_elapsed),
            formatted_eta)
        print_with_overwrite(console_output)

    # Save model for later inference
    checkpoint_path = args.save_path / 'checkpoints' / 'final_model_{}'.format(
        args.iterations)
    model.save(checkpoint_path)

    # Plot losses using moving window of <gap> batches
    losses = [np.mean(losses[window:window + gap])
              for window in np.arange(0, args.iterations, step=gap)]
    plt.plot(np.arange(0, args.iterations, gap), losses)
    plt.legend([model_str])
    plt.title('Cross-entropy loss history for {} network'.format(
        model_str))
    plt.savefig(args.save_path / 'densefs_loss.png')
    print('Finished {}\n\n'.format(model_str))

    # Wise to free these before making a new input_tensor for inference
    del input_tensor
    del gmaker

    # Perform inference on training set if required
    if args.inference_on_training_set:
        inference(
            model, train_types, data_root, args.save_path, args.batch_size,
            labels, autoencoder, resolution=args.resolution,
            dimension=args.dimension, ligmap=args.ligmap, recmap=args.recmap,
            binary_mask=args.binary_mask)

    # Perform inference if test types file is provided
    if test_types is not None:
        inference(
            model, test_types, data_root, args.save_path, args.batch_size,
            labels, autoencoder, resolution=args.resolution,
            dimension=args.dimension, ligmap=args.ligmap, recmap=args.recmap,
            binary_mask=args.binary_mask)


if __name__ == '__main__':
    main()
