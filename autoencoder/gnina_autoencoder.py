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

from autoencoder import autoencoder_definitions, parse_command_line_args
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

    opt_args = {'lr': args.learning_rate, 'loss': args.loss}
    if args.momentum > 0:
        opt_args['momentum'] = args.momentum

    if ae is None:  # No loaded model
        ae = architectures[args.model](
            get_dims(args.dimension, args.resolution, args.ligmap, args.recmap),
            encoding_size=args.encoding_size,
            optimiser=args.optimiser,
            **opt_args)

    ae.summary()

    if args.loss not in ['composite_mse', 'unbalanced_loss']:
        tf.keras.utils.plot_model(
            ae, save_path / 'model.png', show_shapes=True)

    losses, nonzero_losses, zero_losses = train(
        ae, data_root=args.data_root,
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
        binary_mask=args.binary_mask
    )
    print('\nFinished training.')

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
