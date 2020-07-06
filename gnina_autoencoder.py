#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:45:32 2020

@author: scantleb
"""

import argparse
import gnina_embeddings_pb2
import torch
import molgrid
import numpy as np
import os
import pathlib
import time
import tensorflow as tf

from autoencoder import AutoEncoder, DenseAutoEncoder, ShallowAutoEncoder
from collections import defaultdict, deque
from matplotlib import pyplot as plt


def scatter(img, channel=None):
    """Scatter plot with markersize proportional to matrix entry."""
    if channel is not None:
        img = img[channel, :, :, :]
    img = img.squeeze()
    xlin, ylin, zlin = tuple([np.arange(-11.5, 12.5, 0.5)])*3
    x, y, z = np.meshgrid(xlin, ylin, zlin)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=img)


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
        defined in gnina_embeddings.proto. Sturcture is:
            
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
    e = molgrid.ExampleProvider(data_root=data_root, balanced=False,
                                shuffle=False)
    e.populate(types_file)
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
        _, encodings = encoder.predict_on_batch(input_tensor.tonumpy())
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


def main():
    # Parse and sanitise command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", '-r', type=str, required=False,
                        default='')
    parser.add_argument("--train", '-t', type=str, required=True)
    parser.add_argument('--encoding_size', '-e', type=int, required=False,
                        default=50)
    parser.add_argument(
        '--iterations', '-i', type=int, required=False, default=25000)
    parser.add_argument(
        '--save_interval', type=int, required=False, default=10000)
    parser.add_argument(
        '--batch_size', '-b', type=int, required=False, default=16)
    parser.add_argument(
        '--optimiser', '-o', type=str, required=False, default='sgd')
    parser.add_argument(
        '--learning_rate', '-l', type=float, required=False, default=0.01)
    parser.add_argument(
        '--momentum', type=float, required=False, default=None)
    parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.')
    args = parser.parse_args()

    arg_str = '\n'.join(
        ['{0} {1}'.format(arg, getattr(args, arg)) for arg in vars(args)])
    
    optimisers = {
        'sgd' : tf.keras.optimizers.SGD,
        'adadelta' : tf.keras.optimizers.Adadelta,
        'adagrad': tf.keras.optimizers.Adagrad,
        'rmsprop': tf.keras.optimizers.RMSprop
    }

    data_root = os.path.abspath(args.data_root) if len(args.data_root) else ''
    train_types = os.path.abspath(args.train)
    batch_size = args.batch_size
    iterations = args.iterations
    save_interval = args.save_interval
    encoding_size = args.encoding_size
    lr = args.learning_rate
    momentum = args.momentum if args.momentum is not None else 0.0
    savepath = os.path.abspath(
        os.path.join(args.save_path, str(int(time.time()))))
    
    try:
        optimiser = optimisers[args.optimiser.lower()]
    except KeyError:
        raise RuntimeError('{} not a recognised optimiser.'.format(
            args.optimiser))
    if args.momentum is not None and args.optimiser.lower() not in ['sgd',
                                                                    'rmsprop']:
        raise RuntimeError('Momentum only used for RMSProp and SGD optimisers.')
    if not os.path.isfile(args.train):
        raise RuntimeError('{} does not exist.'.format(args.train))
        
    pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(savepath, 'checkpoints')).mkdir(
        parents=True, exist_ok=True)

    slurm_job_id = os.getenv('SLURM_JOB_ID')
    
    arg_str += '\nabsolute_save_path {}\n'.format(os.path.abspath(
            savepath))
    if isinstance(slurm_job_id, str):
        slurm_log_file = os.path.join('~/slurm_logs', 'slurm_{}.out'.format(
            slurm_job_id))
        arg_str += '\nslurm_job_id {0}\nslurm_log_file {1}\n'.format(
            slurm_job_id, slurm_log_file)
    print(arg_str)
    
    with open(os.path.join(savepath, 'config'), 'w') as f:
        f.write(arg_str)
        
    tf.keras.backend.clear_session()

    # Setup libmolgrid to feed Examples into tensorflow objects
    e = molgrid.ExampleProvider(
        data_root=data_root, balanced=False, shuffle=True)
    e.populate(train_types)

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    # Train autoencoder
    zero_losses, nonzero_losses, losses = [], [], []
    ae = DenseAutoEncoder(dims, encoding_size=encoding_size, optimiser=optimiser,
                     lr=lr, momentum=momentum)
    ae.summary()
    tf.keras.utils.plot_model(ae, os.path.join(savepath, 'model.png'),
               show_shapes=True)

    loss_log = 'Iteration Composite Nonzero Zero\n'
    print('Starting training cycle...')
    print('Working directory: {}'.format(os.path.abspath(savepath)))
    loss_ratio = 0.5
    for iteration in range(iterations):
        if iteration == iterations - 1:
            checkpoint_path = os.path.join(
                savepath, 'checkpoints', 'final_model_{}'.format(
                    iteration + 1))
            ae.save_weights(os.path.join(checkpoint_path, 'data'))
        elif not (iteration + 1) % save_interval:
            checkpoint_path = os.path.join(
                savepath, 'checkpoints', 'ckpt_model_{}'.format(
                    iteration + 1))
            ae.save_weights(os.path.join(checkpoint_path, 'data'))
            
        batch = e.next_batch(batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=False)
        max_val = np.amax(input_tensor.tonumpy())
        loss = ae.train_on_batch(
            [input_tensor.tonumpy()/max_val, tf.constant(loss_ratio, shape=(1,))],
            {'reconstruction': input_tensor.tonumpy()/max_val},
            return_dict=True)
        zero_mse = loss['reconstruction_zero_mse']
        nonzero_mse = loss['reconstruction_nonzero_mse']
        if zero_mse > 1e-5:
            loss_ratio = min(50, nonzero_mse / zero_mse)
        else:
            loss_ratio = 50
        loss_str = '{0}\t{1:0.3f}\t{2:0.3f}\t{3:0.3f}'.format(
            iteration, loss['loss'], nonzero_mse, zero_mse)
        
        print(loss_str + '\t{0:0.3f}'.format(loss_ratio))
        loss_log += loss_str + '\n'
        if not iteration % 10:
            with open(os.path.join(savepath, 'loss_log'), 'w') as f:
                f.write(loss_log[:-1])
        
        zero_losses.append(zero_mse)
        nonzero_losses.append(nonzero_mse)
        losses.append(loss['loss'])
    print('Finished training.')

    # Plot zero, nonzero mse
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Batches')
    ax2 = ax1.twinx()
    axes = [ax1, ax2]
    cols = ['r-', 'b-']
    labels = ['Zero_MSE', 'Nonzero_MSE']
    lines = []
    for idx, losses in enumerate([zero_losses, nonzero_losses]):
        gap = 100
        losses = [np.mean(losses[n:n+gap]) for n in range(0, len(losses), gap)]
        line, = axes[idx].plot(
            np.arange(len(losses))*gap, losses, cols[idx], label=labels[idx])
        axes[idx].set_ylabel('Loss')
        lines.append(line)
    ax1.legend(lines, [line.get_label() for line in lines])
    fig.savefig(os.path.join(savepath, 'zero_nonzero_losses.png'))
    
    # Plot composite mse
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Batches')
    axes = [ax1]
    for idx, losses in enumerate([losses]):
        gap = 100
        losses = [np.mean(losses[n:n+gap]) for n in range(0, len(losses), gap)]
        axes[idx].plot(np.arange(len(losses))*gap, losses)
        axes[idx].set_ylabel('Loss')
    ax1.legend(['composite_mse'])
    fig.savefig(os.path.join(savepath, 'composite_loss.png'))
        
    # Save encodings in serialised format
    print('Saving encodings...')
    pathlib.Path(os.path.join(savepath, 'encodings')).mkdir(exist_ok=True,
                                                            parents=True)
    serialised_encodings = calculate_embeddings(
        ae, input_tensor, data_root, train_types)
    for receptor_path, ligands in serialised_encodings.items():
        fname = receptor_path.split('/')[-1].split('.')[0] + '.bin'
        with open(os.path.join(savepath, 'encodings', fname), 'wb') as f:
            f.write(ligands)


if __name__ == '__main__':
    molgrid.set_gpu_enabled(False)
    main()
