"""
Created on Tue Jun 23 14:45:32 2020

@author: scantleb
@brief: Main script for using an autoencoder to reduce the dimensionality of
gnina inputs.
"""

from math import isnan
import molgrid
import numpy as np
import os
import tensorflow as tf
import time

from autoencoder import autoencoder_definitions
from matplotlib import pyplot as plt
from pathlib import Path
from autoencoder.calculate_encodings import calculate_encodings
from tensorflow.python.util import deprecation
from utilities.gnina_functions import Timer, format_time, print_with_overwrite

def main():
    
    # Parse and sanitise command line args
    ae, args = autoencoder_definitions.parse_command_line_args('train')
    
    # There really are a lot of these and they are not useful to scientists
    # using this software. Only log errors (unless verbose)
    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        deprecation._PRINT_DEPRECATION_WARNINGS = False

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

    if ae is None:  # No loaded model
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

    start_time = time.time()
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

        input_tensor_numpy = np.minimum(input_tensor.tonumpy(), 1.0)
        
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

        time_elapsed = time.time() - start_time
        time_per_iter = time_elapsed / (iteration + 1)
        time_remaining = time_per_iter * (args.iterations - iteration - 1)
        formatted_eta = format_time(time_remaining)
        
        if not iteration:
            print('\n')
        
        console_output = ('Iteration: {0}/{1} | loss({2}): {3:0.4f} | ' +
                          'nonzero_mae: {4:0.4f} | zero_mae: {5:0.4f}' +
                          '\nTime elapsed {6} | Time remaining: {7}').format(
            iteration, args.iterations, args.loss, loss['loss'], nonzero_mae,
            zero_mae, format_time(time_elapsed), formatted_eta)
        print_with_overwrite(console_output)
        
        loss_log += loss_str + '\n'
        if not iteration % 10:
            with open(save_path / 'loss_log.txt', 'w') as f:
                f.write(loss_log[:-1])

        zero_losses.append(zero_mae)
        nonzero_losses.append(nonzero_mae)
        losses.append(loss['loss'])
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

    if args.save_encodings: # Save encodings in serialised format
        print('Saving encodings...')
        with Timer() as t:
            calculate_encodings(encoder=ae,
                                gmaker=gmaker,
                                input_tensor=input_tensor,
                                data_root=args.data_root,
                                types_file=args.train,
                                save_path=save_path,
                                rotate=False,
                                verbose=False)
        print('Encodings calculated and saved to {0} in {1} s'.format(
            save_path / 'encodings', t.interval))
            

if __name__ == '__main__':
    main()
