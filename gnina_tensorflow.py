# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import argparse
import os
import time
import torch
import molgrid
import numpy as np
import tensorflow as tf
import pathlib

from model_definitions import define_baseline_model, define_densefs_model

import matplotlib.pyplot as plt

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=str)
parser.add_argument("typesfile", type=str)
parser.add_argument('--test', type=str, required=False)
parser.add_argument('--densefs', '-d', action='store_true')
parser.add_argument(
        '--iterations', '-i', type=int, required=False, default=25000)
parser.add_argument(
        '--batch_size', '-b', type=int, required=False, default=16)
parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.')
args = parser.parse_args()

data_root = os.path.abspath(args.datadir) if args.datadir != 'None' else ''
types_fname = os.path.abspath(args.typesfile)
test_fname = args.test
use_densefs = args.densefs
batch_size = args.batch_size
iterations = args.iterations
savepath = os.path.abspath(
        os.path.join(args.save_path, ['baseline', 'densefs'][use_densefs],
            str(int(time.time()))))

for fname in [types_fname, test_fname]:
    if not os.path.isfile(fname):
        raise RuntimeException('{} does not exist.'.format(fname))

for arg in vars(args):
    print(arg, getattr(args, arg))

gap = 100

e = molgrid.ExampleProvider(data_root=data_root, balanced=True, shuffle=True)
e.populate(types_fname)

"""
print(e.size())
s = ''
print(e.get_type_names()[0])
raise
"""

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(e.num_types())
tensor_shape = (batch_size,) + dims

labels = molgrid.MGrid1f(batch_size)
input_tensor = molgrid.MGrid5f(*tensor_shape)

pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)

if use_densefs:
    print('Using DenseFS for {} iterations'.format(iterations))
    densefs_model = define_densefs_model(dims)
    densefs_losses = []
    losses_string = ''
    for iteration in range(iterations):
        batch = e.next_batch(batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=True)
        batch.extract_label(0, labels)
        loss = densefs_model.train_on_batch(
            input_tensor.tonumpy(), labels.tonumpy())
        densefs_losses.append(float(loss))

        losses_string += '{1} loss: {0:0.3f}\n'.format(loss, iteration)
        with open(os.path.join(
            savepath, 'loss_history_densefs.txt'), 'w') as f:
            f.write(losses_string)

        print(iteration, 'loss: {0:0.3f}'.format(loss))

    densefs_model.save(savepath)

    densefs_losses = [np.mean(densefs_losses[window:window + gap])
            for window in np.arange(0, iterations, step=gap)]
    plt.plot(np.arange(0, iterations, gap), densefs_losses)
    plt.legend(['DenseFS'])
    plt.title('Cross-entropy loss history for DenseFS network')
    plt.savefig(os.path.join(savepath, 'densefs_loss.png'))
    print('Finished DenseFS\n\n')

else:
    print('Using Baseline for {} iterations'.format(iterations))
    baseline_model = define_baseline_model(dims)
    baseline_losses = []
    losses_string = ''
    for iteration in range(iterations):
        batch = e.next_batch(batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=True)
        batch.extract_label(0, labels)
        loss = baseline_model.train_on_batch(
           input_tensor.tonumpy(), labels.tonumpy())
        baseline_losses.append(float(loss))
    
        losses_string += '{1} loss: {0:0.3f}\n'.format(loss, iteration)
        with open(os.path.join(
            savepath, 'loss_history_baseline.txt'), 'w') as f:
            f.write(losses_string)
        baseline_losses.append(float(loss))
        print(iteration, 'loss: {0:0.3f}'.format(loss))

    baseline_model.save(savepath)

    baseline_losses = [np.mean(baseline_losses[window:window + gap])
            for window in np.arange(0, iterations, step=gap)]
    plt.plot(np.arange(0, iterations, gap), baseline_losses)
    plt.legend(['Baseline'])
    plt.savefig(os.path.join(savepath, 'baseline_loss.png'))
    print('Finished Baseline\n\n')


def get_test_info(ep, test_path):
    size = ep.size()
    paths = {}
    with open(test_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            chunks = line.strip().split()
            paths[idx] = (chunks[-2], chunks[-1])
    return paths, size

if test_fname is not None:
    model = densefs_model if use_densefs else baseline_model
    e_test = molgrid.ExampleProvider(
            data_root=data_root, balanced=False, shuffle=False)
    e_test.populate(test_fname)
    paths, size = get_test_info(e_test, test_fname)
    s = ''
    with Timer() as t:
        for iteration in range(size // batch_size):
            batch = e_test.next_batch(batch_size)
            gmaker.forward(batch, input_tensor, 0, random_rotation=False)
            batch.extract_label(0, labels)
            labels_numpy = labels.tonumpy()
            predictions = model.predict_on_batch(
                    input_tensor.tonumpy())
            for i in range(batch_size):
                index = iteration*batch_size + i
                s += '{0} | {1:0.3f} {2} {3}\n'.format(int(labels_numpy[i]),
                        predictions[i][1], paths[index][0], paths[index][1])

        final_batch_size = size % batch_size
        final_tensor_shape = (final_batch_size,) + dims

        batch = e_test.next_batch(final_batch_size)
        final_labels = molgrid.MGrid1f(final_batch_size)
        final_input_tensor = molgrid.MGrid5f(*final_tensor_shape)

        gmaker.forward(batch, final_input_tensor, 0, random_rotation=False)
        batch.extract_label(0, final_labels)
        labels_numpy = final_labels.tonumpy()
        predictions = model.predict_on_batch(final_input_tensor.tonumpy())
        for i in range(final_batch_size):
            index = iteration*batch_size + i
            s += '{0} | {1:0.3f} {2} {3}\n'.format(int(labels_numpy[i]),
                    predictions[i][1], paths[index][0], paths[index][1])

    print('Total inference time:', t.interval, 's')
    modelstr = 'densefs' if use_densefs else 'baseline'
    with open(os.path.join(
        savepath, 'predictions_{}.txt'.format(modelstr)), 'w') as f:
        f.write(s[:-1])
