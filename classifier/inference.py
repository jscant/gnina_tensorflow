"""
Created on Fri Jul 10 14:09:49 2020

@author: scantleb
@brief: Uses trained gnina model to predict class labels for docked structures.

Uses a trained gnina-based keras model to perform inference on a set of
gninatypes gnina inputs, writing the results to disk. The types file is a
text file with lines of the format:
    <label> <receptor_gninatypes_file> <ligand_gninatypes_file>
where <label> is either 0 or 1 (if known) or -1 (if not known), and the
gninatypes files are generated using the gninatyper module of the original
gnina fork (https://github.com/gnina/gnina).
"""

import argparse
import gnina_embeddings_pb2
import torch
import molgrid
from pathlib import Path
import tensorflow as tf

from collections import defaultdict
from gnina_functions import get_test_info, Timer, process_batch
from autoencoder import AutoEncoderBase as ab


def inference(model, test_types, data_root, savepath, batch_size,
              gmaker=None, input_tensor=None, labels=None):
    """Use trained keras model to perform inference on gnina input data.
    
    The model should take a four dimentional tensor as input, with
    (28*48*48*48) recommended.
    
    Arguments:
        model: Trained keras model with 4D input
        test_types: Types file where lines are of the format:
            <label>  <path_to_receptor_gninatype>  <path_to_ligand_gninatype>
        data_root: Directory to which gninatypes files referenced in test_types
            are relative
        savepath: Directory to dump outputs, including predictions and
            serialised embeddings
        batch_size: Number of inputs to perform inference on at once. This
            does not affect predictions, and a large batch size may cause
            out of memory errors.
        gmaker: molgrid.Gmaker object. If None, one will be created.
        input_tensor: molgrid.MGrid5f object. If none, one will be created
            with dimensions inferred from the inputs.
        labels: molgrid.MGrid1f object where predictions are stored for each
            batch.      
    """
    savepath = Path(savepath).resolve() # in case this is a string
    test_types_stem = Path(test_types).stem
    predictions_fname = 'predictions_{}.txt'.format(test_types_stem)
    predictions_fname = savepath / predictions_fname
    
    # Setup molgrid.ExampleProvider and GridMaker to feed into network
    e_test = molgrid.ExampleProvider(
        data_root=data_root, balanced=False, shuffle=False)
    e_test.populate(test_types)

    paths, size = get_test_info(test_types)  # For indexing in output

    if gmaker is None:
        gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e_test.num_types())
    tensor_shape = (batch_size,) + dims

    if input_tensor is None:
        input_tensor = molgrid.MGrid5f(*tensor_shape)

    if labels is None:
        labels = molgrid.MGrid1f(batch_size)
        
    # Make output directories
    embeddings_dir = savepath / 'encodings_{}'.format(test_types_stem)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print('Performing inference on {} examples'.format(size))
    representations_dict = defaultdict(dict)
    labels_dict = defaultdict(dict)
    test_output_string = ''
    with open(predictions_fname, 'w') as f:
        f.write('')
        
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
                labels_dict[rec_path][lig_path] = labels_numpy[i]
                test_output_string += '{0} | {1:0.7f} {2} {3}\n'.format(
                    int(labels_numpy[i]),
                    predictions[i][1],
                    paths[index][0],
                    paths[index][1]
                )
            if not iteration % 1000:
                with open(predictions_fname, 'a') as f:
                    f.write(test_output_string)
                test_output_string = ''

        remainder = size % batch_size            
        labels_numpy, predictions = process_batch(
            model, e_test, gmaker, input_tensor, labels_tensor=labels,
            train=False)
        representations = [p.flatten() for p in predictions[1]]
        predictions = predictions[0]
        for i in range(remainder):
            index = size - (size % batch_size) + i
            rec_path = paths[index][0]
            lig_path = paths[index][1]
            representations_dict[
                rec_path][lig_path] = representations[i]
            labels_dict[rec_path][lig_path] = labels_numpy[i]
            test_output_string += '{0} | {1:0.7f} {2} {3}\n'.format(
                int(labels_numpy[i]),
                predictions[i][1],
                paths[index][0],
                paths[index][1]
            )
        with open(predictions_fname, 'a') as f:
            f.write(test_output_string[:-1])


    print('Total inference time:', t.interval, 's')
    
    # Save predictions to disk
    serialised_embeddings = {}
    for receptor_path, ligands in representations_dict.items():
        receptor_msg = gnina_embeddings_pb2.protein()
        receptor_msg.path = receptor_path
        for ligand_path, representation in ligands.items():
            label = int(labels_dict[receptor_path][ligand_path])
            ligand_msg = receptor_msg.ligand.add()
            ligand_msg.path = ligand_path
            ligand_msg.embedding.extend(representation)
            ligand_msg.label = label
        serialised_embeddings[receptor_path] = receptor_msg.SerializeToString()    
    
    for receptor_path, ligands in serialised_embeddings.items():
        fname = Path(receptor_path).stem + '.bin'
        with open(embeddings_dir / fname, 'wb') as f:
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
    
    model = tf.keras.models.load_model(
        args.model_dir,
        custom_objects={
            'zero_mse': ab.zero_mse,
            'nonzero_mse': ab.nonzero_mse,
            'composite_mse': ab.composite_mse
            }
    )
    inference(
        model, args.test, args.data_root, args.save_path, args.batch_size)