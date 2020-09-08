"""
Created on Tue Jun 23 14:45:32 2020

@author: scantleb
@brief: Main script for using an autoencoder to reduce the dimensionality of
gnina inputs.
"""

import os
from pathlib import Path

import molgrid
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.util import deprecation

from autoencoder import autoencoder_definitions, parse_command_line_args, \
    schedules
from autoencoder.calculate_encodings import calculate_encodings
from autoencoder.train import train
from utilities.gnina_functions import Timer, get_dims


def main():
    # Parse and sanitise command line args
    ae, args = parse_command_line_args.parse_command_line_args('train')

    # There really are a lot of these and they are not useful to scientists
    # using this software. Only log errors (unless verbose)
    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        deprecation._PRINT_DEPRECATION_WARNINGS = False

    # For use later when defining model
    architectures = {'single': autoencoder_definitions.SingleLayerAutoEncoder,
                     'dense': autoencoder_definitions.DenseAutoEncoder}

    molgrid.set_gpu_enabled(1 - args.use_cpu)
    arg_str = '\n'.join(
        ['{0} {1}'.format(arg, getattr(args, arg)) for arg in vars(args)])

    save_path = Path(args.save_path, args.name).expanduser().resolve()

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

    # Use learning rate schedule or single learning rate
    if args.max_lr > 0 and args.min_lr > 0:
        if args.optimiser not in ['sgdw', 'adamw']:
            raise RuntimeError(
                'Learning rate scheduling only compatible with AdamW and SGDW '
                'optimisers.'
            )
        if args.learning_rate_schedule is None:
            raise RuntimeError(
                'Max and min learning rates must be used in conjunction with '
                'a learning rate schedule.'
            )

        lrs_args = [args.min_lr, args.max_lr]
        lrs_kwargs = {}
        if args.learning_rate_schedule == '1cycle':
            scheduler = schedules.OneCycle
            lrs_kwargs.update({'iterations': args.iterations})
        elif args.learning_rate_schedule == 'warm_restarts':
            scheduler = schedules.WarmRestartCosine
            lrs_kwargs.update(
                {'beta': args.warm_beta, 'period': args.warm_period})
        lrs = scheduler(*lrs_args, **lrs_kwargs)

        opt_args = {'weight_decay': 1e-4}
    else:
        opt_args = {'lr': args.learning_rate}
        lrs = None

    if args.momentum > 0:
        opt_args['momentum'] = args.momentum

    if ae is None:  # No loaded model
        ae = architectures[args.model](
            get_dims(args.dimension, args.resolution, args.ligmap, args.recmap),
            encoding_size=args.encoding_size,
            optimiser=args.optimiser,
            loss=args.loss,
            hidden_activation=args.hidden_activation,
            final_activation=args.final_activation,
            **opt_args)

    ae.summary()

    if args.loss != 'composite_mse':
        tf.keras.utils.plot_model(
            ae, save_path / 'model.png', show_shapes=True)

    losses, nonzero_losses, zero_losses = train(
        ae,
        data_root=args.data_root,
        train_types=args.train,
        iterations=args.iterations,
        batch_size=args.batch_size,
        save_path=save_path,
        dimension=args.dimension,
        resolution=args.resolution,
        loss_fn=args.loss,
        ligmap=args.ligmap,
        recmap=args.recmap,
        save_interval=args.save_interval,
        overwrite_checkpoints=args.overwrite_checkpoints,
        binary_mask=args.binary_mask,
        lrs=lrs
    )
    print('\nFinished training.')

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
        losses = [np.mean(losses[n:n + gap]) for n in
                  range(0, len(losses), gap)]
        line, = axes[idx].plot(
            np.arange(len(losses)) * gap, losses, cols[idx], label=labels[idx])
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
        losses = [np.mean(losses[n:n + gap]) for n in
                  range(0, len(losses), gap)]
        axes[idx].plot(np.arange(len(losses)) * gap, losses)
        axes[idx].set_ylabel('Loss')
    ax1.legend([args.loss])
    fig.savefig(save_path / 'composite_loss.png')

    if args.save_encodings:  # Save encodings in serialised format
        print('Saving encodings...')
        with Timer() as t:
            calculate_encodings(encoder=ae,
                                data_root=args.data_root,
                                batch_size=args.batch_size,
                                types_file=args.train,
                                save_path=save_path,
                                dimension=args.dimension,
                                resolution=args.resolution,
                                ligmap=args.ligmap,
                                recmap=args.recmap,
                                rotate=False,
                                binary_mask=args.binary_mask
                                )
        print('Encodings calculated and saved to {0} in {1} s'.format(
            save_path / 'encodings', t.interval))


if __name__ == '__main__':
    main()
