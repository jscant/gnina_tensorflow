"""
Created on Tue Jun 23 14:45:32 2020

@author: scantleb
@brief: Main script for using an autoencoder to reduce the dimensionality of
gnina inputs.
"""

from pathlib import Path

import molgrid
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

from autoencoder import parse_command_line_args, \
    schedules
from autoencoder.calculate_encodings import calculate_encodings
from autoencoder.train import train
from utilities.gnina_functions import Timer, get_dims, load_autoencoder


def main():
    # Parse and sanitise command line args
    ae_class, args = parse_command_line_args.parse_command_line_args('train')

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

    arg_str = '\n'.join(
        ['{0} {1}'.format(param, argument) for param, argument
         in vars(args).items() if param not in barred_args]
    )

    save_path = Path(args.save_path, args.name).expanduser().resolve()

    if args.momentum > 0 and \
            args.optimiser.lower() not in ['sgd', 'rmsprop', 'sgdw']:
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

    # tf.keras.backend.clear_session()

    input_dims = get_dims(
        args.dimension, args.resolution, args.ligmap, args.recmap)
    ae = load_autoencoder(args, args.load_model, input_dims,
                          lrs, opt_args)

    if args.load_model is not None:
        ae(np.random.rand(args.batch_size, *input_dims).astype('float32'))
        ae.load_weights(args.load_model)

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
        starting_iter = int(str(
            Path(args.load_model).name).split('_')[-1].split('.')[0])
        with open(log_fname, 'r') as f:
            loss_log = '\n'.join(
                f.read().split('\n')[:starting_iter + 1]) + '\n'
        barred_args.append('load_model')

    with open(save_path / 'model.summary', 'w') as f:
        ae.summary(line_length=80, print_fn=lambda x: f.write(x + '\n'))
    ae.summary()
    ae.plot(save_path / 'model.png', show_shapes=True)

    losses, nonzero_losses, zero_losses = train(
        ae,
        iterations=args.iterations,
        save_path=save_path,
        loss_fn=args.loss,
        save_interval=args.save_interval,
        overwrite_checkpoints=args.overwrite_checkpoints,
        loss_log=loss_log,
        starting_iter=starting_iter,
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
