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

from model_definitions import define_baseline_model, define_densefs_model
from gnina_functions import process_batch, beautify_config

from tensorflow.keras.utils import plot_model
from inference import inference

import matplotlib.pyplot as plt

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
    checkpoint_path = os.path.join(
        savepath, 'checkpoints', 'ckpt_model_{}'.format(
            iteration + 1))
    model.save(checkpoint_path)

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
        inference(model, test_types, data_root, savepath, batch_size,
                  gmaker, input_tensor, labels)
        

if __name__ == '__main__':
    main()
