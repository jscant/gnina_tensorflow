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
import tensorflow_addons as tfa
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
                     'dense': autoencoder_definitions.DenseAutoEncoder,
                     'multi': autoencoder_definitions.MultiLayerAutoEncoder,
                     'res': autoencoder_definitions.ResidualAutoEncoder}

    molgrid.set_gpu_enabled(1 - args.use_cpu)

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
                {'beta': args.lrs_beta, 'period': args.lrs_period})
        elif args.learning_rate_schedule == 'stepwise':
            scheduler = schedules.StepWiseDecay
            lrs_kwargs.update(
                {'t': args.lrs_period, 'beta': args.lrs_beta})
        else:
            raise RuntimeError(
                'learning_rate_schedule must be one of "1cycle", '
                '"warm_restarts" or "stepwise".')
        lrs = scheduler(*lrs_args, **lrs_kwargs)
        opt_args = {'weight_decay': 1e-4}
    else:
        opt_args = {'lr': args.learning_rate}
        lrs = schedules.ConstantLearningRateSchedule(args.learning_rate)

    if args.momentum > 0:
        opt_args['momentum'] = args.momentum
        if args.optimiser.startswith('sgd'):
            opt_args['nesterov'] = args.nesterov

    barred_args = ['resume']
    loss_log = None
    starting_iter = 0
    if args.resume:
        if not args.load_model:
            raise RuntimeError(
                '--resume must be used in conjunction with load_model')
        if args.optimiser == 'adamw':
            optimiser = tfa.optimizers.AdamW
        elif args.optimiser == 'sgdw':
            optimiser = tfa.optimizers.SGDW
        else:
            optimiser = tf.keras.optimizers.get(args.optimiser).__class__
        ae.optimizer = optimiser(
            **opt_args
        )
        log_fname = Path(
            args.load_model).expanduser().parents[1] / 'loss_log.txt'
        starting_iter = int(str(Path(args.load_model).name).split('_')[-1])
        with open(log_fname, 'r') as f:
            loss_log = '\n'.join(
                f.read().split('\n')[:starting_iter + 1]) + '\n'
        barred_args.append('load_model')

    arg_str = '\n'.join(
        [
            '{0} {1}'.format(param, argument)
            for param, argument
            in vars(args).items()
            if param not in barred_args
        ]
    )

    save_path = Path(args.save_path, args.name).expanduser().resolve()

    if args.momentum > 0 and args.optimiser.lower() not in ['sgd', 'rmsprop',
                                                            'sgdw']:
        raise RuntimeError(
            'Momentum only used for RMSProp and SGD optimisers.')
    if not Path(args.train).exists():
        raise RuntimeError('{} does not exist.'.format(args.train))

    Path(save_path, 'checkpoints').mkdir(parents=True, exist_ok=True)

    arg_str += '\nabsolute_save_path {}\n'.format(save_path)
    print(arg_str)

    if not args.resume:
        with open(save_path / 'config', 'w') as f:
            f.write(arg_str)

    tf.keras.backend.clear_session()

    if ae is None:  # No loaded model
        ae = architectures[args.model](
            get_dims(args.dimension, args.resolution, args.ligmap, args.recmap),
            encoding_size=args.encoding_size,
            optimiser=args.optimiser,
            loss=args.loss,
            hidden_activation=args.hidden_activation,
            final_activation=args.final_activation,
            encoding_activation=args.encoding_activation,
            metric_distance_threshold=args.metric_distance_threshold,
            learning_rate_schedule=lrs,
            adversarial=args.adversarial,
            adversarial_variance=args.adversarial_variance,
            **opt_args)
    else:
        ae.learning_rate_schedule = lrs

    with open(save_path / 'model.summary', 'w') as f:
        ae.summary(line_length=80, print_fn=lambda x: f.write(x + '\n'))
    ae.summary()

    if args.loss not in ['composite_mse', 'distance_mse']:
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
        metric_distance_threshold=args.metric_distance_threshold,
        overwrite_checkpoints=args.overwrite_checkpoints,
        binary_mask=args.binary_mask,
        denoising=args.denoising,
        loss_log=loss_log,
        starting_iter=starting_iter
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
