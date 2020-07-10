# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:46:06 2020

@author: scantleb
@brief: Main script for training and inference with gnina-based neural
networks (https://github.com/gnina/gnina) .

Requirements: libmolgrid, pytorch (1.3.1), tensorflow 2.x
"""

import argparse
import os
import time
import torch
import molgrid
import numpy as np
import pathlib
import datetime

from collections import defaultdict
from model_definitions import define_baseline_model, define_densefs_model
from tensorflow.keras.utils import plot_model
import gnina_embeddings_pb2

import matplotlib.pyplot as plt


class Timer:
    """
    Simple timer class.

    To time a block of code, wrap it like so:

        with Timer() as t:
            <some_code>
        total_time = t.interval

    The time taken for the code to execute is stored in t.interval.
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def beautify_config(config, fname=None):
    """
    Formats dictionary into two columns, sorted in alphabetical order.

    Arguments:
        config: any dictionary
        fname: string containing valid path to filename for writing beautified
            config to. If left blank, output will be printed to stdout.
    """
    output = 'Time of experiment generation: {:%d-%m-%Y %H:%M:%S}\n\n'.format(
        datetime.datetime.now())

    sorted_config = sorted(
        [(arg, value) for arg, value in config.items()], key=lambda x: x[0]
    )

    # Padding magic
    max_len = max([len(i) for i in [arg for arg, _ in sorted_config]]) + 4
    padding = ' ' * max_len

    for arg, value in sorted_config:
        output += '{0}:{1}{2}\n'.format(arg, padding[len(arg):], value)
    if fname is not None:
        with open(fname, 'w') as f:
            f.write(output[:-1])
    else:
        print(output)


def get_test_info(test_file):
    """
    Obtains information about gninatypes file.

    Arguments:
        test_file: text file containing labels and paths to gninatypes files

    Returns:
        dictionary containing tuples with the format:
            {index : (receptor_path, ligand_path)}
            where index is the line number, receptor_path is the path to the
            receptor gninatype and ligand_path is the path to the ligand
            gninatype.
    """
    paths = {}
    with open(test_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            chunks = line.strip().split()
            paths[idx] = (chunks[-2], chunks[-1])
    return paths, len(paths)


def process_batch(model, example_provider, gmaker, input_tensor,
                  labels_tensor=None, train=True):
    """
    Feeds forward and backpropagates (if train==True) batch of examples.

    Arguments:
        model: compiled tensorflow model
        example_provider: molgrid.ExampleProvider object populated with a
            types file
        gmaker: molgrid.GridMaker object
        input_tensor: molgrid.MGrid<x>f object, where <x> is the dimentions
            of the input (including a dimention for batch size)
        labels_tensor: molgrid.MGrid1f object, for storing true labels. If
            labels_tensor is None and train is False, , return value will be
            a vector of predictions.

    Returns:
        if labels_tensor is None and train is False: numpy.ndarray of
            predictions
        if labels_tensor is specified and train is False: tuple containing
            numpy.ndarray of labels and numpy.ndarray of predictions
        if labels_tensor is specified and train is True: float containing
            the mean loss over the batch (usually cross-entropy)

    Raises:
        RuntimeError: if labels_tensor is None and train is True
    """
    if train and labels_tensor is None:
        raise RuntimeError('Labels must be provided for backpropagation',
                           'if train == True')
    batch = example_provider.next_batch(input_tensor.shape[0])
    gmaker.forward(batch, input_tensor, 0, random_rotation=train)

    if labels_tensor is None:  # We don't know labels; just return predictions
        return model.predict_on_batch(input_tensor.tonumpy())

    batch.extract_label(0, labels_tensor)  # y_true
    if train:  # Return loss
        return model.train_on_batch(
            input_tensor.tonumpy(), labels_tensor.tonumpy())
    else:  # Return labels, predictions
        return (labels_tensor.tonumpy(),
                model.predict_on_batch(input_tensor.tonumpy()))


def main():
    # Create and parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=False,
                        default=pathlib.Path.home())
    parser.add_argument("--train", type=str, required=False)
    parser.add_argument('--test', type=str, required=False)
    parser.add_argument('--densefs', '-d', action='store_true')
    parser.add_argument(
        '--iterations', '-i', type=int, required=False, default=25000)
    parser.add_argument(
        '--batch_size', '-b', type=int, required=False, default=16)
    parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.')
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument(
    '--use_cpu', '-g', action='store_true')
    parser.add_argument(
        '--use_densenet_bc', action='store_true')
    args = parser.parse_args()

    # We need to train or test
    if not (args.train or args.test):
        raise RuntimeError('Please specify at least one of --train or '
                           '--test')
    else:
        # Check if types files exist
        for fname in [types for types in [args.train, args.test]
                      if types is not None]:
            if not os.path.isfile(fname):
                raise RuntimeError('{} does not exist.'.format(fname))

    config_args = {}

    data_root = os.path.abspath(args.data_root)
    train_types = os.path.abspath(args.train)
    test_types = args.test
    use_densefs = args.densefs
    batch_size = args.batch_size
    iterations = args.iterations
    savepath = os.path.abspath(
        os.path.join(args.save_path, ['baseline', 'densefs'][use_densefs],
                     str(int(time.time()))))
    save_interval = args.save_interval

    # Create config dict for saving to disk later
    config_args['data_root'] = data_root
    config_args['train'] = train_types if train_types else 'NONE'
    config_args['test'] = test_types if test_types else 'NONE'
    config_args['model'] = ['baseline', 'densefs'][use_densefs]
    config_args['batch_size'] = batch_size
    config_args['iterations'] = iterations
    config_args['save_path'] = savepath
    config_args['save_interval'] = save_interval
    config_args['use_cpu'] = args.use_cpu
    config_args['use_densenet_bc'] = args.use_densenet_bc
    beautify_config(config_args)
    
    if args.use_cpu:
        molgrid.set_gpu_enabled(False)
    else:
        molgrid.set_gpu_enabled(True)

    gap = 100  # Window to average training loss over (in batches)

    # Setup libmolgrid to feed Examples into tensorflow objects
    e = molgrid.ExampleProvider(
        data_root=data_root, balanced=True, shuffle=True)
    e.populate(train_types)

    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(e.num_types())
    tensor_shape = (batch_size,) + dims

    labels = molgrid.MGrid1f(batch_size)
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    pathlib.Path(os.path.join(savepath, 'checkpoints')).mkdir(
        parents=True, exist_ok=True)

    # We are ready to define our model and train
    losses = []
    if use_densefs:
        model = define_densefs_model(dims, bc=args.use_densenet_bc)
    else:
        model = define_baseline_model(dims)

    model_str = ['Baseline', 'DenseFS'][use_densefs]
    plot_model(model, os.path.join(savepath, 'model.png'),
               show_shapes=True)
    beautify_config(config_args, os.path.join(savepath, 'config'))
    model.summary()

    losses_string = ''
    for iteration in range(iterations):
        if iteration == iterations - 1:
            checkpoint_path = os.path.join(
                savepath, 'checkpoints', 'final_model_{}'.format(
                    iteration + 1))
            model.save(checkpoint_path)
        elif not (iteration + 1) % save_interval:
            checkpoint_path = os.path.join(
                savepath, 'checkpoints', 'ckpt_model_{}'.format(
                    iteration + 1))
            model.save(checkpoint_path)
        # Data: e > gmaker > input_tensor > network (forward and backward pass)
        loss = process_batch(model, e, gmaker, input_tensor, labels,
                             train=True)

        # Save losses to disk
        if not isinstance(loss, float):
            loss = loss[0]
        losses.append(loss)
        losses_string += '{1} loss: {0:0.3f}\n'.format(loss, iteration)
        with open(os.path.join(savepath, 'loss_history_{}.txt'.format(
                model_str.lower())), 'w') as f:
            f.write(losses_string)
        print(iteration, 'loss: {0:0.3f}'.format(loss))

    # Save model for later inference
    model.save(savepath)

    # Plot losses using moving window of <gap> batches
    losses = [np.mean(losses[window:window + gap])
              for window in np.arange(0, iterations, step=gap)]
    plt.plot(np.arange(0, iterations, gap), losses)
    plt.legend([model_str])
    plt.title('Cross-entropy loss history for {} network'.format(
        model_str))
    plt.savefig(os.path.join(savepath, 'densefs_loss.png'))
    print('Finished {}\n\n'.format(model_str))

    # Perform inference if test types file is provided
    if test_types is not None:

        # Setup molgrid.ExampleProvider and GridMaker to feed into network
        e_test = molgrid.ExampleProvider(
            data_root=data_root, balanced=False, shuffle=False)
        e_test.populate(test_types)

        paths, size = get_test_info(test_types)  # For indexing in output
        test_output_string = ''

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

        print('Total inference time ({}):'.format(model_str), t.interval, 's')
        
        # Save predictions to disk
        with open(os.path.join(savepath, 'predictions_{}.txt'.format(
                model_str.lower())), 'w') as f:
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
            break
        
        pathlib.Path(os.path.join(savepath, 'encodings')).mkdir(
            parents=True, exist_ok=True)
        for receptor_path, ligands in serialised_embeddings.items():
            fname = receptor_path.split('/')[-1].split('.')[0] + '.bin'
            with open(os.path.join(savepath, 'encodings', fname), 'wb') as f:
                f.write(ligands)

if __name__ == '__main__':
    main()
