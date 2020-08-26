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
import molgrid
import tensorflow as tf

from collections import defaultdict
from pathlib import Path
from utilities.gnina_functions import get_test_info, Timer, process_batch, \
    print_with_overwrite


def inference(model, test_types, data_root, savepath, batch_size,
              gmaker=None, input_tensor=None, labels=None, autoencoder=None,
              dimension=23.0, resolution=0.5, ligmap=None, recmap=None):
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
        autoencoder: object derived from AutoEncoderBase virtual class; all
            inputs will first be encoded and then decoded using the
            autoencoder so inference will be on the reconstruction.
        dimension: (if gmaker is not provided) size of box around ligand in
            Angstroms
        resolution: (if gmaker is not provided) resolution of voxelisation
            in Angstroms
        ligmap: Text file containing definitions of ligand input channels
        recmap: Text file containing definitions of receptor input channels
    """
    savepath = Path(savepath).resolve() # in case this is a string
    test_types_stem = Path(test_types).stem
    predictions_fname = 'predictions_{}.txt'.format(test_types_stem)
    predictions_fname = savepath / predictions_fname
    
    # Setup molgrid.ExampleProvider and GridMaker to feed into network
    
    if recmap is not None and ligmap is not None:
        rec_typer = molgrid.FileMappedGninaTyper(recmap)
        lig_typer = molgrid.FileMappedGninaTyper(ligmap)
        e_test = molgrid.ExampleProvider(
            rec_typer, lig_typer, data_root=str(data_root), balanced=False,
            shuffle=False)
    else:
        e_test = molgrid.ExampleProvider(
            data_root=str(data_root), balanced=False, shuffle=False)
        
    e_test.populate(str(test_types))

    paths, size = get_test_info(test_types)  # For indexing in output

    if gmaker is None:
        gmaker = molgrid.GridMaker(dimension=dimension, resolution=resolution)
    dims = gmaker.grid_dimensions(e_test.num_types())
    tensor_shape = (batch_size,) + dims

    if input_tensor is None:
        input_tensor = molgrid.MGrid5f(*tensor_shape)

    if labels is None:
        labels = molgrid.MGrid1f(batch_size)
        
    # Make output directory
    savepath.mkdir(parents=True, exist_ok=True)

    print('Performing inference on {} examples'.format(size))
    labels_dict = defaultdict(dict)
    test_output_string = ''
    with open(predictions_fname, 'w') as f:
        f.write('')
        
    iterations = size // batch_size
    with Timer() as t:
        for iteration in range(iterations):
            labels_numpy, predictions = process_batch(
                model, e_test, gmaker, input_tensor, labels_tensor=labels, 
                train=False, autoencoder=autoencoder)
            for i in range(batch_size):
                index = iteration*batch_size + i
                rec_path = paths[index][0]
                lig_path = paths[index][1]
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
            print_with_overwrite('Iteration: {0}/{1}'.format(
                iteration + 1, iterations + int(size % batch_size)))

        remainder = size % batch_size            
        labels_numpy, predictions = process_batch(
            model, e_test, gmaker, input_tensor, labels_tensor=labels,
            train=False, autoencoder=autoencoder)
        for i in range(remainder):
            index = size - (size % batch_size) + i
            rec_path = paths[index][0]
            lig_path = paths[index][1]
            labels_dict[rec_path][lig_path] = labels_numpy[i]
            test_output_string += '{0} | {1:0.7f} {2} {3}\n'.format(
                int(labels_numpy[i]),
                predictions[i][1],
                paths[index][0],
                paths[index][1]
            )
        with open(predictions_fname, 'a') as f:
            f.write(test_output_string[:-1])
        print_with_overwrite('Iteration: {0}/{1}'.format(
            iterations + int(size % batch_size),
            iterations + int(size % batch_size)))

    print('Total inference time:', t.interval, 's')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('test', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--densefs', '-d', action='store_true')
    parser.add_argument('--save_path', '-s', type=str, default='')
    parser.add_argument('--use_cpu', '-c', action='store_true')
    args = parser.parse_args()
    
    # Load some parameters from orginal model config
    config = Path(args.model_dir).parents[1] / 'config'
    recmap, ligmap = None, None
    gridmaker_args = { 'dimension': None, 'resolution': None }
    with open(config, 'r') as f:
        for line in f.readlines():
            chunks = line.strip().split()
            if len(chunks):
                if chunks[0].startswith('recmap'):
                    recmap = chunks[1]
                elif chunks[0].startswith('ligmap'):
                    ligmap = chunks[1]
                elif chunks[0].startswith('dimension'):
                    gridmaker_args['dimension'] = float(chunks[1])
                elif chunks[0].startswith('resolution'):
                    gridmaker_args['resolution'] = float(chunks[1])
                    
    gridmaker_args = {i: j for i, j in gridmaker_args.items() if j is not None}
    
    tf.keras.backend.clear_session()
    
    molgrid.set_gpu_enabled(1-int(args.use_cpu))

    # Setup libmolgrid to feed Examples into tensorflow objects
    if recmap is not None and ligmap is not None:
        rec_typer = molgrid.FileMappedGninaTyper(recmap)
        lig_typer = molgrid.FileMappedGninaTyper(ligmap)
        e = molgrid.ExampleProvider(
            rec_typer, lig_typer, data_root=str(args.data_root),
            balanced=False, shuffle=True)
    else:
        e = molgrid.ExampleProvider(
            data_root=str(args.data_root), balanced=False, shuffle=True)
    e.populate(str(args.test))

    gmaker = molgrid.GridMaker(**gridmaker_args)
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (args.batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    model = tf.keras.models.load_model(
        args.model_dir,
    )
    inference(
        model, args.test, args.data_root, args.save_path, args.batch_size,
        gmaker=gmaker, input_tensor=input_tensor, ligmap=ligmap, recmap=recmap)