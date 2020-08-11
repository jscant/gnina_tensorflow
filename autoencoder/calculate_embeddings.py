#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:11:57 2020

@author: scantleb
@brief: Use trained autoencoder to calculate embeddings for gnina inputs.
"""

import torch
import molgrid
import tensorflow as tf

from autoencoder import autoencoder_definitions
from collections import defaultdict, deque
from pathlib import Path
from utilities import gnina_embeddings_pb2, gnina_functions


def calculate_embeddings(encoder, gmaker, input_tensor, data_root, types_file,
                         save_path, rotate=False):
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
        defined in gnina_embeddings.proto. The structure is:

            {receptor_path : serialised_protobuf_messages (1 per ligand)}
    """

    def get_paths():
        """Reads types file to give path and indexing information."""
        paths = defaultdict(deque)
        with open(types_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                chunks = line.strip().split()
                paths[chunks[-2]].append((idx, int(chunks[0]), chunks[-1]))
        return paths

    # Setup for gnina
    batch_size = input_tensor.shape[0]
    e = molgrid.ExampleProvider(data_root=str(data_root), balanced=False,
                                shuffle=False)
    e.populate(str(types_file))

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
            embeddings[global_idx] = encodings[batch_idx, :]
            
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
            label = ligand[1]
            ligand_path = ligand[2]
            embedding = embeddings[global_idx]
            ligand_msg = receptor_msg.ligand.add()
            ligand_msg.path = ligand_path
            ligand_msg.embedding.extend(embedding)
            ligand_msg.label = label
        serialised_embeddings[receptor_path] = receptor_msg.SerializeToString()

    encodings_dir = Path(save_path) / 'encodings'
    encodings_dir.mkdir(exist_ok=True, parents=True)
    for receptor_path, ligands in serialised_embeddings.items():
        fname = receptor_path.split('/')[-1].split('.')[0] + '.bin'
        with open(encodings_dir / fname, 'wb') as f:
            f.write(ligands)

if __name__ == '__main__':
    # Parse and sanitise command line args
    autoencoder, args = autoencoder_definitions.parse_command_line_args('test')
    autoencoder.summary()
    
    molgrid.set_gpu_enabled(1 - args.use_cpu)

    tf.keras.backend.clear_session()

    # Setup libmolgrid to feed Examples into tensorflow objects
    e = molgrid.ExampleProvider(
        data_root=str(args.data_root), balanced=False, shuffle=False)
    e.populate(str(args.test))
    
    gmaker = molgrid.GridMaker(
        binary=args.binary_mask,
        dimension=args.dimensions,
        resolution=args.resolution)

    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (args.batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)
    
    with gnina_functions.Timer() as t:
        calculate_embeddings(
            autoencoder, gmaker, input_tensor, args.data_root, args.test,
            save_path=args.save_path, rotate=False)
    print('Inference took {} s'.format(t.interval))
    print('Encodings written to {}'.format(args.save_path))
    
    
    
    