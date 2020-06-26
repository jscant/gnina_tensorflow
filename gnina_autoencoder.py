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

from autoencoder import AutoEncoder
from collections import defaultdict, deque
from matplotlib import pyplot as plt


def scatter(img, channel=None):
    if channel is not None:
        img = img[channel, :, :, :]
    img = img.squeeze()
    xlin, ylin, zlin = np.arange(-11.5, 12.5, 0.5), np.arange(-11.5, 12.5, 0.5), np.arange(-11.5, 12.5, 0.5)
    x, y, z = np.meshgrid(xlin, ylin, zlin)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, s=img)


def calculate_embeddings(encoder, input_tensor, data_root, types_file, rotate=False):
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
        Serialised protobuf message with structure defined in
        gnina_embeddings.proto
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
    batch_size = 16  # This doesn't really matter (aside from speed)
    e = molgrid.ExampleProvider(data_root=data_root, balanced=False,
                                shuffle=False)
    e.populate(types_file)
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types())

    # Need a dictionary mapping {rec : deque([(idx, lig), ...])} where idx
    # is the position of the receptor/ligand pair in the types file
    paths = get_paths()
    size = sum([len(info) for _, info in paths.items()])

    iterations = size // batch_size

    embeddings = {}
    gnina_embeddings = gnina_embeddings_pb2.database()

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
        receptor_msg = gnina_embeddings.protein.add()
        receptor_msg.protein_path = receptor_path
        for ligand in ligands:
            global_idx = ligand[0]
            ligand_path = ligand[1]
            embedding = embeddings[global_idx]
            ligand_msg = receptor_msg.ligand.add()
            ligand_msg.ligand_path = ligand_path
            ligand_msg.embedding.extend(embedding)

    return gnina_embeddings.SerializeToString()


def main():
    # Parse and sanitise command line args
    molgrid.set_gpu_enabled(False)
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
        '--save_path', '-s', type=str, required=False, default='.')
    args = parser.parse_args()

    if not os.path.isfile(args.train):
        raise RuntimeError('{} does not exist.'.format(args.train))
        
    arg_str = '\n'.join(
        ['{0} {1}'.format(arg, getattr(args, arg)) for arg in vars(args)])
    print(arg_str)

    data_root = os.path.abspath(args.data_root) if len(args.data_root) else ''
    train_types = os.path.abspath(args.train)
    batch_size = args.batch_size
    iterations = args.iterations
    save_interval = args.save_interval
    encoding_size = args.encoding_size
    savepath = os.path.abspath(
        os.path.join(args.save_path, str(int(time.time()))))
    pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(savepath, 'checkpoints')).mkdir(
        parents=True, exist_ok=True)
    
    with open(os.path.join(savepath, 'config'), 'w') as f:
        f.write(arg_str)

    # Setup libmolgrid to feed Examples into tensorflow objects
    e = molgrid.ExampleProvider(
        data_root=data_root, balanced=False, shuffle=True)
    e.populate(train_types)

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    # Train autoencoder
    losses = []
    ae = AutoEncoder(dims, encoding_size=encoding_size)
    ae.summary()
    
    print('Starting training cycle...')
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
        gmaker.forward(batch, input_tensor, 0, random_rotation=True)
        loss = ae.train_on_batch(
            input_tensor.tonumpy(),
            {'reconstruction': input_tensor.tonumpy()},
            return_dict=True)
        print('mse:', loss['loss'], 'relative_mse:', loss['reconstruction_nonzero_mse'])
        losses.append(loss['loss'])
    print('Finished training.')
    
    # Plot loss
    gap = 100
    losses = [np.mean(losses[n:n+gap]) for n in range(0, len(losses), gap)]
    plt.plot(np.arange(len(losses))*gap, losses)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(savepath, 'loss.png'))

    # Save encodings in serialised format
    print('Saving encodings...')
    serialised_encodings = calculate_embeddings(ae, input_tensor, data_root, train_types)
    with open(os.path.join(savepath, 'serialised_encodings.bin'), 'wb') as f:
        f.write(serialised_encodings)

    


if __name__ == '__main__':
    x, bins = main()
