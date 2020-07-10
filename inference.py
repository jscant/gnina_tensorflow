#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:09:49 2020

@author: scantleb
"""

import argparse
import os
import torch
import molgrid
import pathlib

from collections import defaultdict

from gnina_functions import get_test_info, Timer, process_batch
import gnina_embeddings_pb2


def inference(model, test_types, data_root, savepath, batch_size,
              gmaker=None, input_tensor=None, labels=None):
    
    # Setup molgrid.ExampleProvider and GridMaker to feed into network
    e_test = molgrid.ExampleProvider(
        data_root=data_root, balanced=False, shuffle=False)
    e_test.populate(test_types)

    paths, size = get_test_info(test_types)  # For indexing in output
    test_output_string = ''

    if gmaker is None:
        gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e_test.num_types())
    tensor_shape = (batch_size,) + dims

    if input_tensor is None:
        input_tensor = molgrid.MGrid5f(*tensor_shape)

    if labels is None:
        labels = molgrid.MGrid1f(batch_size)

    print('Performing inference on {} examples'.format(size))
    representations_dict = defaultdict(dict)
    with Timer() as t:
        for iteration in range(size // batch_size):
            labels_numpy, predictions = process_batch(
                model, e_test, gmaker, input_tensor, labels_tensor=labels, 
                train=False)
            representations = [p.flatten() for p in predictions[1]]
            predictions = predictions[0]
            for i in range(batch_size):
                index = iteration*batch_size + i
                rec_path = paths[index][0]
                lig_path = paths[index][1]
                representations_dict[
                    rec_path][lig_path] = representations[i]
                test_output_string += '{0} | {1:0.3f} {2} {3}\n'.format(
                    int(labels_numpy[i]),
                    predictions[i][1],
                    paths[index][0],
                    paths[index][1]
                )

        remainder = size % batch_size            
        labels_numpy, predictions = process_batch(
            model, e_test, gmaker, input_tensor, labels_tensor=labels,
            train=False)
        representations = [p.flatten() for p in predictions[1]]
        predictions = predictions[0]
        for i in range(remainder):
            index = size // batch_size + i
            rec_path = paths[index][0]
            lig_path = paths[index][1]
            representations_dict[
                rec_path][lig_path] = representations[i]
            test_output_string += '{0} | {1:0.3f} {2} {3}\n'.format(
                int(labels_numpy[i]),
                predictions[i][1],
                paths[index][0],
                paths[index][1]
            )

    print('Total inference time:', t.interval, 's')
    
    # Save predictions to disk
    pathlib.Path(os.path.join(savepath, 'encodings')).mkdir(
        parents=True, exist_ok=True)
    with open(os.path.join(savepath, 'predictions.txt'), 'w') as f:
        f.write(test_output_string[:-1])
    
    serialised_embeddings = {}
    for receptor_path, ligands in representations_dict.items():
        receptor_msg = gnina_embeddings_pb2.protein()
        receptor_msg.path = receptor_path
        for ligand_path, representation in ligands.items():
            ligand_msg = receptor_msg.ligand.add()
            ligand_msg.path = ligand_path
            ligand_msg.embedding.extend(representation)
        serialised_embeddings[receptor_path] = receptor_msg.SerializeToString()    
    
    for receptor_path, ligands in serialised_embeddings.items():
        fname = receptor_path.split('/')[-1].split('.')[0] + '.bin'
        with open(os.path.join(savepath, 'encodings', fname), 'wb') as f:
            f.write(ligands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('test', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--densefs', '-d', action='store_true')
    parser.add_argument('--save_path', '-s', type=str, default='')
    args = parser.parse_args()
    
    if args.densefs:  
        from model_definitions import define_densefs_model as model_def
    else:
        from model_definitions import define_baseline_model as model_def
    model = model_def((28, 48, 48, 48))
    inference(
        model, args.test, args.data_root, args.save_path, args.iterations)