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
from collections import defaultdict
from pathlib import Path

import molgrid
import tensorflow as tf

from autoencoder.autoencoder_definitions import zero_mse, nonzero_mse, \
    composite_mse, nonzero_mae, zero_mae, trimmed_nonzero_mae, trimmed_zero_mae, \
    close_mae, close_nonzero_mae, close_zero_mae
from utilities.gnina_functions import get_test_info, Timer, process_batch, \
    print_with_overwrite


def inference(model, test_types, data_root, savepath, batch_size, labels=None,
              autoencoder=None, dimension=23.0, resolution=0.5, ligmap=None,
              recmap=None, binary_mask=False):
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
        binary_mask: Inputs are either in {0, 1} rather than non-negative real
    """
    savepath = Path(savepath).resolve()  # in case this is a string
    test_types_stem = Path(test_types).stem
    predictions_fname = 'predictions_{}.txt'.format(test_types_stem)
    predictions_fname = savepath / predictions_fname

    # Setup molgrid.ExampleProvider and GridMaker to feed into network

    example_provider_kwargs = {
        'data_root': str(Path(data_root).expanduser()), 'balanced': False,
        'shuffle': False, 'cache_structs': False
    }
    if recmap is not None and ligmap is not None:
        rec_typer = molgrid.FileMappedGninaTyper(recmap)
        lig_typer = molgrid.FileMappedGninaTyper(ligmap)
        e_test = molgrid.ExampleProvider(
            rec_typer, lig_typer, **example_provider_kwargs)
    else:
        e_test = molgrid.ExampleProvider(**example_provider_kwargs)

    e_test.populate(str(test_types))

    paths, size = get_test_info(test_types)  # For indexing in output

    gmaker = molgrid.GridMaker(
        dimension=dimension, resolution=resolution, binary=binary_mask)
    dims = gmaker.grid_dimensions(e_test.num_types())
    tensor_shape = (batch_size,) + dims

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
            tf.keras.backend.clear_session()
            labels_numpy, predictions = process_batch(
                model, e_test, gmaker, input_tensor, labels_tensor=labels,
                train=False, autoencoder=None)
            for i in range(batch_size):
                index = iteration * batch_size + i
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
    parser.add_argument('--save_path', '-s', type=str, default='')
    parser.add_argument('--use_cpu', '-c', action='store_true')
    args = parser.parse_args()

    # Load some parameters from orginal model config
    config = Path(args.model_dir).parents[1] / 'config'
    recmap, ligmap, autoencoder_path = None, None, None
    binary_mask = False
    with open(config, 'r') as f:
        for line in f.readlines():
            chunks = line.strip().split()
            if len(chunks) == 2:
                param = chunks[0]
                value = chunks[1]
                if param == 'recmap':
                    recmap = value
                elif param == 'ligmap':
                    ligmap = value
                elif param == 'dimension':
                    dimension = float(value)
                elif param == 'resolution':
                    resolution = float(value)
                elif param == 'binary_mask':
                    if value == 'True':
                        binary_mask = True
                elif param == 'autoencoder':
                    autoencoder_path = value

    molgrid.set_gpu_enabled(1 - int(args.use_cpu))
    custom_objects = {
        'zero_mse': zero_mse,
        'nonzero_mse': nonzero_mse,
        'composite_mse': composite_mse,
        'nonzero_mae': nonzero_mae,
        'zero_mae': zero_mae,
        'trimmed_nonzero_mae': trimmed_nonzero_mae,
        'trimmed_zero_mae': trimmed_zero_mae,
        'close_mae': close_mae,
        'close_nonzero_mae': close_nonzero_mae,
        'close_zero_mae': close_zero_mae
    }

    model = tf.keras.models.load_model(
        args.model_dir,
        custom_objects=custom_objects
    )

    if autoencoder_path is not None:
        autoencoder = tf.keras.models.load_model(
            autoencoder_path,
            custom_objects=custom_objects
        )
    else:
        autoencoder = None

    inference(
        model, args.test, args.data_root, args.save_path, args.batch_size,
        dimension=dimension, resolution=resolution, ligmap=ligmap,
        recmap=recmap, binary_mask=binary_mask, autoencoder=autoencoder)
