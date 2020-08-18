#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:56:42 2020

@author: scantleb
@brief: Unit tests for calculate_encodings.py
"""

import tensorflow as tf
import torch
import molgrid
import numpy as np
import pytest

from autoencoder.calculate_encodings import calculate_encodings
from autoencoder import autoencoder_definitions
from collections import defaultdict
from pathlib import Path
from utilities import gnina_embeddings_pb2 as ge


def wipe_directory(directory):
    """Recursively removes all items in a directory, then the directory itself.
    
    Arguments:
        directory: Location of directory to wipe (str or PosixPath)
    """
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            wipe_directory(item)
        else:
            item.unlink()
    directory.rmdir()


def test_calculate_encodings():
    autoencoder = tf.keras.models.load_model(
        'test/trained_autoencoder',
        custom_objects={
            'zero_mse': autoencoder_definitions.zero_mse,
            'nonzero_mse': autoencoder_definitions.nonzero_mse,
            'composite_mse': autoencoder_definitions.composite_mse,
            'nonzero_mae': autoencoder_definitions.nonzero_mae,
            'zero_mae': autoencoder_definitions.zero_mae,
            'approx_heaviside': autoencoder_definitions.approx_heaviside,
            'unbalanced_loss': autoencoder_definitions.unbalanced_loss,
        }
    )
    
    batch_size = 16
    temporary_directory = Path('test/tmp_save_path')
    test_types = 'data/small_chembl_test.types'
    
    tf.keras.backend.clear_session()

    # Setup libmolgrid to feed Examples into tensorflow objects
    e = molgrid.ExampleProvider(
        data_root='data', balanced=False, shuffle=False)
    e.populate(test_types)

    gmaker = molgrid.GridMaker(dimension=18.0, resolution=1.0)

    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    calculate_encodings(
        autoencoder, gmaker, input_tensor, 'data',
        test_types, save_path=temporary_directory,
        rotate=False)
    
    labels = defaultdict(lambda: defaultdict(lambda: None))
    with open(test_types, 'r') as f:
        for line in f.readlines():
            chunks = line.split()
            label = int(chunks[0])
            rec = chunks[1]
            lig = chunks[2]
            labels[rec][lig] = label
    s = ''
    for filename in Path(temporary_directory, 'encodings').glob('*.bin'):
        encodings = ge.protein()
        encodings.ParseFromString(open(filename, 'rb').read())
        rec_path = encodings.path
        for ligand_struct in encodings.ligand:
            label = ligand_struct.label
            ligand_path = ligand_struct.path
            assert np.array(ligand_struct.embedding).shape == (500, )
            assert labels[rec_path][ligand_path] == label
            
    wipe_directory(temporary_directory)
    
    
test_calculate_encodings()
