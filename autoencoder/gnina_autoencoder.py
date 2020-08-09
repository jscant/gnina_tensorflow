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

from autoencoder.autoencoder_definitions import SingleLayerAutoEncoder as AE
from collections import defaultdict, deque
from matplotlib import pyplot as plt
from pathlib import Path
from utilities import gnina_embeddings_pb2


def calculate_embeddings(encoder, input_tensor, data_root, types_file,
                         rotate=False):
    """Calculates encodings for gnina inputs.

    Uses trained AutoEncoder object to calculate the embeddings of all gnina
    inputs in types_file, which are then serialised to a protobuf message.

    Arguments:
        encoder: trained AutoEncoder object
        data_root: path to which all paths in types file are relative
        types_file: contains list of gninatypes paths, storing the gnina-style
            inputs which are to be encoded in a lower dimensional space
        rotate: whether to randomly rotate gnina inputs in increments of 15
            degrees

    Returns:
        Dictionary of serialised protein protobuf messages with structure
        defined in gnina_embeddings.proto. Structure is:

            {receptor_path : serialised_protobuf_messages (1 per ligand)}
    """

    def get_paths():
        """Reads types file to give path and indexing information"""
        paths = defaultdict(deque)
        with open(types_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                chunks = line.strip().split()
                paths[chunks[-2]].append((idx, chunks[-1]))
        return paths

    # Setup for gnina
    batch_size = input_tensor.shape[0]
    e = molgrid.ExampleProvider(data_root=str(data_root), balanced=False,
                                shuffle=True)
    e.populate(str(types_file))
    gmaker = molgrid.GridMaker()

    # Need a dictionary mapping {rec : deque([(idx, lig), ...])} where idx
    # is the position of the receptor/ligand pair in the types file
    paths = get_paths()
    size = sum([len(info) for _, info in paths.items()])

    iterations = size // batch_size

    embeddings = {}
    serialised_embeddings = {}

    # Inference (obtain encodings)
    for iteration in range(iterations):
        batch = e.next_batch(batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=rotate)
        encodings = encoder.predict_on_batch(input_tensor.tonumpy())
        for batch_idx in range(batch_size):
            global_idx = iteration * batch_size + batch_idx
            embeddings[global_idx] = encodings[batch_idx]

    remainder = size % batch_size
    batch = e.next_batch(batch_size)
    gmaker.forward(batch, input_tensor, 0, random_rotation=rotate)
    _, encodings = encoder.predict_on_batch(input_tensor.tonumpy())

    for batch_idx in range(remainder):
        global_idx = iterations * batch_size + batch_idx
        embeddings[global_idx] = encodings[batch_idx]

    # We have a dictionary of encodings, each with a unique global index. This
    # index maps onto the original paths dictionary so we can create a message
    # per protein.
    for receptor_path, ligands in paths.items():
        receptor_msg = gnina_embeddings_pb2.protein()
        receptor_msg.path = receptor_path
        for ligand in ligands:
            global_idx = ligand[0]
            ligand_path = ligand[1]
            embedding = embeddings[global_idx]
            ligand_msg = receptor_msg.ligand.add()
            ligand_msg.path = ligand_path
            ligand_msg.embedding.extend(embedding)
        serialised_embeddings[receptor_path] = receptor_msg.SerializeToString()

    return serialised_embeddings


def pickup(path, autoencoder_class):
    """Loads saved autoencoder.

    Arguments:
        path: location of saved weights and architecture
        ae_class: class derived from the AutoEncoderBase class

    Returns:
        DenseAutoEncoder object initialised with weights from saved checkpoint,
        as well as a dictionary containing the command line arguments taken
        from the config file used to produce the saved model.
    """
    path = Path(path).resolve()
    config_path = path.parents[2]
    config_file = config_path / 'config'
    if not config_file.exists():
        raise RuntimeError(
            "No config file found in experiment's base directory ({})".format(
                config_path))
    args = defaultdict(str)
    with open(config_file, 'r') as f:
        for line in f.readlines():
            chunks = line.split()
            if len(chunks) < 2:
                continue
            key = chunks[0]
            value = chunks[1]
            if value.lower() == 'none':
                args[key] = None
                continue
            if value.lower() in ['true', 'false']:
                args[key] = [False, True][value.lower() == 'true']
                continue
            try:
                if len(value.split('.')) > 1:
                    true_value = float(value)
                else:
                    true_value = int(value)
            except ValueError:
                true_value = value
            args[key] = true_value
    ae = tf.keras.models.load_model(
        path,
        custom_objects={
            'zero_mse': autoencoder_class.zero_mse,
            'nonzero_mse': autoencoder_class.nonzero_mse,
            'composite_mse': autoencoder_class.composite_mse,
            'nonzero_mae': autoencoder_class.nonzero_mae,
            'zero_mae': autoencoder_class.zero_mae,
            'approx_heaviside': autoencoder_class.approx_heaviside,
            'unbalanced_loss': autoencoder_class.unbalanced_loss,
        }
    )
    # Bug with add_loss puts empty dict at the end of model._layers which
    # interferes with some functionality (such as
    # tf.keras.utils.plot_model)
    ae._layers = [layer for layer in ae._layers if isinstance(
        layer, tf.keras.layers.Layer)]
    return ae, args


def main():
    # Parse and sanitise command line args
    parser = argparse.ArgumentParser()
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
        '--optimiser', '-o', type=str, required=False, default='sgd')
    parser.add_argument(
        '--learning_rate', '-l', type=float, required=False)
    parser.add_argument(
        '--momentum', type=float, required=False, default=None)
    parser.add_argument(
        '--loss', type=str, required=False, default='mse')
    parser.add_argument(
        '--final_activation', type=str, required=False, default='sigmoid')
    parser.add_argument('--binary_mask', action='store_true')
    parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.')
    parser.add_argument(
        '--pickup', '-p', type=str, required=False,
        help='Pick up from a checkpoint; use same config as that checkpoint' +
        'was generated by')
    parser.add_argument(
        '--use_cpu', '-g', action='store_true')
    args = parser.parse_args()

    load_model = bool(args.pickup) # False if empty string or NoneType
    if load_model:
        ae, loaded_args = pickup(args.pickup, AE)
        args = argparse.Namespace(
            pickup=args.pickup,
            data_root=loaded_args['data_root'],
            train=loaded_args['train'],
            batch_size=loaded_args['batch_size'],
            iterations=args.iterations if isinstance(args.iterations, int)
            else loaded_args['iterations'],
            save_interval=loaded_args['save_interval'],
            encoding_size=loaded_args['encoding_size'],
            learning_rate=args.learning_rate if
            isinstance(args.learning_rate, float) else
            loaded_args['learning_rate'],
            momentum=args.momentum if isinstance(args.momentum, float)
            else loaded_args['momentum'],
            optimiser=args.optimiser if isinstance(args.optimiser, str)
            else loaded_args['optimiser'],
            save_path=args.save_path,
            use_cpu=args.use_cpu,
            loss=loaded_args.get('loss', 'mse'),
            binary_mask=loaded_args.get('binary_mask', False),
            final_activation=loaded_args.get('final_activation', 'unknown')
        )

    molgrid.set_gpu_enabled(1-args.use_cpu)
    arg_str = '\n'.join(
        ['{0} {1}'.format(arg, getattr(args, arg)) for arg in vars(args)])

    optimisers = {
        'sgd': tf.keras.optimizers.SGD,
        'adadelta': tf.keras.optimizers.Adadelta,
        'adagrad': tf.keras.optimizers.Adagrad,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'adamax': tf.keras.optimizers.Adamax,
        'adam': tf.keras.optimizers.Adam
    }

    data_root = Path(args.data_root).resolve() if len(
        args.data_root) else ''
    train_types = Path(args.train).resolve()
    batch_size = args.batch_size
    iterations = args.iterations
    save_interval = args.save_interval
    encoding_size = args.encoding_size
    loss_fn = args.loss
    lr = args.learning_rate
    momentum = args.momentum if args.momentum is not None else 0.0

    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if isinstance(slurm_job_id, str):
        slurm_log_file = Path.home() / 'slurm_{}.out'.format(slurm_job_id)
        arg_str += '\nslurm_job_id {0}\nslurm_log_file {1}\n'.format(
            slurm_job_id, slurm_log_file)
        savepath = Path(args.save_path, slurm_job_id).resolve()
    else:
        savepath = Path(args.save_path, str(int(time.time()))).resolve()

    try:
        optimiser = optimisers[args.optimiser.lower()]
    except KeyError:
        raise RuntimeError('{} not a recognised optimiser.'.format(
            args.optimiser))
    if (args.momentum is not None and
            args.optimiser.lower() not in ['sgd', 'rmsprop']):
        raise RuntimeError(
            'Momentum only used for RMSProp and SGD optimisers.')
    if not Path(args.train).exists():
        raise RuntimeError('{} does not exist.'.format(args.train))

    Path(savepath, 'checkpoints').mkdir(parents=True, exist_ok=True)

    arg_str += '\nabsolute_save_path {}\n'.format(savepath)
    print(arg_str)

    with open(savepath / 'config', 'w') as f:
        f.write(arg_str)

    tf.keras.backend.clear_session()

    # Setup libmolgrid to feed Examples into tensorflow objects
    e = molgrid.ExampleProvider(
        data_root=str(data_root), balanced=False, shuffle=True)
    e.populate(str(train_types))

    for n in e.get_type_names():
        print(n)

    gmaker = molgrid.GridMaker(binary=args.binary_mask, dimension=18.0,
                               resolution=1.0)
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    # Train autoencoder
    zero_losses, nonzero_losses, losses = [], [], []

    opt_args = {'lr': lr, 'loss': loss_fn}
    if momentum > 0:
        opt_args['momentum'] = momentum

    if not load_model:
        ae = AE(dims, encoding_size=encoding_size,
                optimiser=optimiser, **opt_args)
    ae.summary()
    tf.keras.utils.plot_model(ae, savepath / 'model.png', show_shapes=True)

    loss_ratio = 0.5
    loss_log = 'iteration {} nonzero_mae zero_mae nonzero_mean\n'.format(
        loss_fn)
    print('Starting training cycle...')
    print('Working directory: {}'.format(savepath))

    for iteration in range(iterations):
        if not (iteration + 1) % save_interval and iteration < iterations - 1:
            checkpoint_path = Path(
                savepath,
                'checkpoints',
                'ckpt_model_{}'.format(iteration + 1)
            )
            ae.save(checkpoint_path)

        batch = e.next_batch(batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=False)

        input_tensor_numpy = input_tensor.tonumpy()

        mean_nonzero = np.mean(
            input_tensor_numpy[np.where(input_tensor_numpy > 0)])

        x_inputs = {'input_image': input_tensor_numpy}
        if loss_fn == 'composite_mse':
            x_inputs['frac'] = tf.constant(loss_ratio, shape=(1,))

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
            with open(savepath / 'loss_log.txt', 'w') as f:
                f.write(loss_log[:-1])

        zero_losses.append(zero_mae)
        nonzero_losses.append(nonzero_mae)
        losses.append(loss['loss'])
    print('Finished training.')

    # Save final trained autoencoder
    checkpoint_path = Path(
        savepath, 'checkpoints', 'final_model_{}'.format(iterations))
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
    fig.savefig(savepath / 'zero_nonzero_losses.png')

    # Plot composite mse
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Batches')
    axes = [ax1]
    for idx, losses in enumerate([losses]):
        gap = 100
        losses = [np.mean(losses[n:n+gap]) for n in range(0, len(losses), gap)]
        axes[idx].plot(np.arange(len(losses))*gap, losses)
        axes[idx].set_ylabel('Loss')
    ax1.legend([loss_fn])
    fig.savefig(savepath / 'composite_loss.png')

    # Save encodings in serialised format
    print('Saving encodings...')
    encodings_dir = savepath / 'encodings'
    encodings_dir.mkdir(exist_ok=True, parents=True)
    serialised_encodings = calculate_embeddings(
        ae, input_tensor, data_root, train_types)
    for receptor_path, ligands in serialised_encodings.items():
        fname = receptor_path.split('/')[-1].split('.')[0] + '.bin'
        with open(encodings_dir / fname, 'wb') as f:
            f.write(ligands)


if __name__ == '__main__':
    main()
