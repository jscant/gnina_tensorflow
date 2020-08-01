"""
Created on Fri Jul 31 15:10:59 2020

@author: scantleb
@brief: Unit tests for gnina_functions.py. Still under construction.
"""

import torch
import molgrid
import numpy as np

from utilities.gnina_functions import get_test_info, process_batch
from classifier.model_definitions import define_baseline_model, define_densefs_model

def test_get_test_info():
    test_fname = 'data/small_chembl_test.types'
    paths, size = get_test_info(test_fname)
    assert size == 510
    with open(test_fname, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            assert paths[idx][0] == line.split()[1]
            assert paths[idx][1] == line.split()[2]
    
def test_process_batch():

    train_types = 'data/small_chembl_test.types'
    batch_size = 1
    
    e = molgrid.ExampleProvider(
        data_root='data', balanced=False, shuffle=False)
    e.populate(train_types)

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,) + dims

    labels = molgrid.MGrid1f(batch_size)
    input_tensor = molgrid.MGrid5f(*tensor_shape)
    
    model = define_densefs_model(dims)    
    
    res = process_batch(model, e, gmaker, input_tensor, labels, train=False)
    assert len(res) == 510
    assert isinstance(res, np.ndarray)