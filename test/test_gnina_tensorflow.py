"""
Created on Fri Jul 31 15:10:59 2020

@author: scantleb
@brief: Unit tests for gnina_functions.py. Still under construction.
"""

import torch
import molgrid
import numpy as np
import pytest

from utilities.gnina_functions import get_test_info, process_batch
from classifier.model_definitions import define_baseline_model

def test_get_test_info():
    """Unit test for get_test_info
    
    Correct size of data set obtained.
    Indices match with line numbers in types file.
    """
    test_fname = 'data/small_chembl_test.types'
    paths, size = get_test_info(test_fname)
    assert size == 510
    with open(test_fname, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            assert paths[idx][0] == line.split()[1]
            assert paths[idx][1] == line.split()[2]
    
def test_process_batch():
    """Unit test for process_batch
    
    Class probabilities returned should sum to 1.
    Number of labels, predictions and embeddings should equal batch_size.
    Predictions should have the shape (batch_size, 2).
    """
    train_types = 'data/small_chembl_test.types'
    batch_size = 4
    
    e = molgrid.ExampleProvider(
        data_root='data', balanced=False, shuffle=False)
    e.populate(train_types)

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,) + dims

    labels = molgrid.MGrid1f(batch_size)
    input_tensor = molgrid.MGrid5f(*tensor_shape)
    
    model = define_baseline_model(dims)    
    
    y_true, outputs = process_batch(
        model, e, gmaker, input_tensor, labels, train=False)
    
    y_pred, final_layer = outputs
    
    assert len(y_true) == 4
    assert y_pred.shape == (batch_size, 2)
    n_embeddings, _ = final_layer.shape
    assert n_embeddings == batch_size
    
    y_pred_sums = np.sum(y_pred, axis=1)
    
    assert y_pred_sums == pytest.approx(1.0)