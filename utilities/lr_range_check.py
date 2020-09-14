#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 19:30:43 2020

@author: scantleb
@brief: LR-Range test for autoencoder.

The LR-Range test aims to find the optimum learning rate, or range of
learning rates, for any given model. Models are trained from a cold start
at different rates, the loss is plotted, and the three regions are identified:
too large, too small, and 'just right'. 'Just right' is the region where
the loss after n iterations is decreasing with increasing learning rate. The
boundaries of this region are the ideal minimum and maximum learning rates
for cyclical learning rate schedules and 1cycle.

Example usage:

python3 lr_range_test.py dense 1e-7 1e-1 -dr data -t \
        data/small_chembl_test.types -o sgd --loss mse

(Paper at https://arxiv.org/abs/1506.01186)
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from autoencoder import autoencoder_definitions, train
from utilities.gnina_functions import get_dims


class LRRangeTest:
    """Class for performing the LR-Range test on autoencoder models."""

    def __init__(self, model: tf.keras.Model, ae_args:dict, train_types: str,
                 data_root: str, dimension: float, resolution: float,
                 batch_size: int, loss_fn: str, ligmap: str, recmap: str,
                 binary_mask: bool):
        """Initialise lr range test.

        Arguments:
            model: autoencoder model class definition derived from
                AutoEncoderBase
            ae_args: dict containing all arguments for the constructor of the
                autoencoder
            train_types: types file containing training data
            data_root: path relative to which paths in train_types are taken
            dimension: length of side of cube in which ligand is situated in
                Angstroms
            resolution: resolution of voxelisation in Angstroms
            batch_size: number of examples per training batch
            loss_fn: loss function for autoencoder
            ligmap: text file containing ligand channel information
            recmap: text file containing receptor channel information
            binary_mask: inputs are either 0 or 1, depending on if they are 0 or
                > 0 originally
        """
        self.train_types = train_types
        self.data_root = data_root
        self.dimension = dimension
        self.resolution = resolution
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.ligmap = ligmap
        self.recmap = recmap
        self.binary_mask = binary_mask
        self.model_class = model

        self.dims = ae_args['dims']
        del ae_args['dims']
        self.ae_kwargs = ae_args

        m = model(self.dims, **self.ae_kwargs)
        self.initial_weights = m.get_weights()
        del m

    def run_model(self, lr: float, iterations: int):
        """Train model.

        Initialise weights and learning rate and train for a set number of
        iterations

        Arguments:
            lr: learning rate for optimiser
            iterations: number of batches to train over

        Returns:
            Three lists containing the loss history, the mean average arror
            over inputs equal to zero, and the mean average error over inputs
            greater than zero.
        """

        # Let's not build multiple memory-intensive graphs
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        print('\nLearning rate:\n', lr)

        self.ae_kwargs['learning_rate'] = lr
        model = self.model_class(self.dims, **self.ae_kwargs)
        model.set_weights(self.initial_weights)

        losses, nonzero_mae, zero_mae = train.train(
            model,
            data_root=self.data_root,
            train_types=self.train_types,
            iterations=iterations,
            batch_size=self.batch_size,
            save_path=None,
            dimension=self.dimension,
            resolution=self.resolution,
            loss_fn=self.loss_fn,
            ligmap=self.ligmap,
            recmap=self.recmap,
            save_interval=-1,
            binary_mask=self.binary_mask,
            silent=False)

        # More memory required due to leaky TensorFlow methods
        del model

        return losses, zero_mae, nonzero_mae

    def plot_results(self, hist_len, save_name='lr_range_test.png'):
        """Plot results of LR-range test.

        Arguments:
            hist_len: how many batches at the end to take the mean loss from
            save_name: filename for LR-range test results graph
        """
        fig, (loss_ax, z_ax, nz_ax) = plt.subplots(1, 3, figsize=(18, 6))
        axes = (loss_ax, z_ax, nz_ax)
        titles = ('Loss {}'.format(self.loss_fn), 'Nonzero MAE', 'Zero MAE')
        for idx, ax in enumerate(axes):
            ax.set_xscale('log', basex=10)
            lr_arr = []
            loss_arr = []
            for res in self.results:
                lr_arr.append(res[0])
                loss_arr.append(np.mean(res[1][idx][-hist_len:]))
            ax.plot(lr_arr, loss_arr)
            ax.set_title(titles[idx])
        fig.savefig(Path(save_name).expanduser())

    def __call__(self, min_lr, max_lr, iters, hist_len=50,
                 save_name='lr_range_test.png'):
        """Run LR-range test.

        Arguments:
            min_lr: lowest learning rate to test
            max_lr: highest learning rate to test
            iters: number of iterations to train autoencoder before calculating
                mean end loss
            hist_len: number of iterations at the end over which to take the
                mean end loss
            save_name: filename to which graph of results is written
        """
        ceiling = max_lr
        curr = min_lr
        lrs = []
        self.results = []
        while curr <= ceiling:
            lrs.append(curr)
            curr = np.round(curr + 10 ** (np.floor(np.log10(curr))), 10)
        for lr in lrs:
            losses = self.run_model(lr, iters)
            self.results.append((lr, losses))
        self.plot_results(hist_len, save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='One of either "single", "multi", "dense" or '
                             '"res", indicating autoencoder architecture')
    parser.add_argument('min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('max_lr', type=float, help='Maximum learning rate')
    parser.add_argument('--data_root', '-dr',
                        help='Path relative to which paths in types file are '
                             'taken')
    parser.add_argument('--train_types', '-t',
                        help='Types file containing label, receptor path and '
                             'ligand path information')
    parser.add_argument('--opt', '-o', type=str, required=True,
                        help='Optimiser to use')
    parser.add_argument('--momentum', type=float, required=False, default=-1.0,
                        help='Momentum (for SGD/RMSprop optimisers)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use Nesterov momentum')
    parser.add_argument('--loss', type=str, required=True, help='Loss function')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        required=False,
                        help='Number of examples per batch')
    parser.add_argument('--encoding_size', '-e', type=int, required=True,
                        help='Size of code in autoencoder')
    parser.add_argument('--iterations', '-i', type=int, required=False,
                        default=1000, help='Number of iterations to run each '
                                           'model for at each learning rate')
    parser.add_argument('--dimension', '-d', type=float, required=False,
                        default=23.0,
                        help='Length of side of cube in which ligand is '
                             'situated in Angstroms')
    parser.add_argument('--resolution', '-r', type=float, required=False,
                        default=1.0,
                        help='Resolution of voxelisation in Angstroms')
    parser.add_argument('--hidden_activation', type=str, required=False,
                        default='sigmoid', help='Activation function for '
                                                'hidden layers')
    parser.add_argument('--final_activation', type=str, required=False,
                        default='sigmoid', help='Activation function for '
                                                'reconstruction layer')
    parser.add_argument('--ligmap', '-l', type=str, required=False,
                        default=None, help='Text file containing ligand channel'
                                           ' information')
    parser.add_argument('--recmap', '-m', type=str, required=False,
                        default=None, help='Text file containing receptor '
                                           'channel information')
    parser.add_argument('--hist_len', '-hl', type=int, required=False,
                        default=50,
                        help='How many batches at the end to take the mean '
                             'loss from')
    parser.add_argument('--save_filename', '-s', type=str, required=False,
                        default='lr_test.png',
                        help='Filename for results graph')
    parser.add_argument('--binary_mask', action='store_true',
                        help='Inputs are either 0 or 1, depending on if they '
                             'are 0 or > 0 originally')

    args = parser.parse_args()
    dims = get_dims(args.dimension, args.resolution, args.ligmap, args.recmap)

    autoencoder_class = {
        'single': autoencoder_definitions.SingleLayerAutoEncoder,
        'dense': autoencoder_definitions.DenseAutoEncoder,
        'multi': autoencoder_definitions.MultiLayerAutoEncoder,
        'res': autoencoder_definitions.ResidualAutoEncoder}[args.model]
    opt_args = {}
    if args.momentum > 0:
        opt_args['momentum'] = args.momentum
    if args.nesterov:
        opt_args['nesterov'] = True

    # We have to construct a new model every cycle from the input args because
    # repeated tf.keras.Model.compile cause severe memory leaks, even when
    # the current graph is reset and the session is cleared.
    autoencoder_args = {
        'dims': dims,
        'encoding_size': args.encoding_size,
        'loss': args.loss,
        'hidden_activation': args.hidden_activation,
        'final_activation': args.final_activation,
        'optimiser': args.opt
    }
    if args.momentum > 0:
        autoencoder_args.update({'momentum': args.momentum})
    if args.nesterov:
        autoencoder_args.update({'nesterov': True})

    range_test = LRRangeTest(
        autoencoder_class, autoencoder_args, args.train_types,
        data_root=args.data_root, dimension=args.dimension,
        resolution=args.resolution, batch_size=args.batch_size,
        loss_fn=args.loss, ligmap=args.ligmap, recmap=args.recmap,
        binary_mask=args.binary_mask)
    range_test(min_lr=args.min_lr, max_lr=args.max_lr, iters=args.iterations,
               hist_len=args.hist_len, save_name=args.save_filename)
    print('Results saved to {}'.format(
        Path(args.save_filename).expanduser().resolve()))
