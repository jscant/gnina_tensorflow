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
import pathlib
import datetime

from model_definitions import define_baseline_model, define_densefs_model
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def beautify_config(config, fname=None):
    output = 'Time of experiment generation: {:%d-%m-%Y %H:%M:%S}\n\n'.format(
        datetime.datetime.now())
    sorted_config = sorted([(arg, value)
                            for arg, value in config.items()], key=lambda x: x[0])
    max_len = max([len(i) for i in [arg for arg, _ in sorted_config]]) + 4
    padding = ' ' * max_len
    for arg, value in sorted_config:
        output += '{0}:{1}{2}\n'.format(arg, padding[len(arg):], value)
    if fname is not None:
        with open(fname, 'w') as f:
            f.write(output[:-1])
    else:
        print(output)


def get_test_info(ep, test_path):
    size = ep.size()
    paths = {}
    with open(test_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            chunks = line.strip().split()
            paths[idx] = (chunks[-2], chunks[-1])
    return paths, size


parser = argparse.ArgumentParser()
parser.add_argument("data_root", type=str)
parser.add_argument("train", type=str)
parser.add_argument('--test', type=str, required=False)
parser.add_argument('--densefs', '-d', action='store_true')
parser.add_argument(
    '--iterations', '-i', type=int, required=False, default=25000)
parser.add_argument(
    '--batch_size', '-b', type=int, required=False, default=16)
parser.add_argument(
    '--save_path', '-s', type=str, required=False, default='.')
args = parser.parse_args()

config_args = {}

data_root = os.path.abspath(args.data_root) if args.data_root != 'None' else ''
train_types = os.path.abspath(args.train)
test_types = args.test
use_densefs = args.densefs
batch_size = args.batch_size
iterations = args.iterations
savepath = os.path.abspath(
    os.path.join(args.save_path, ['baseline', 'densefs'][use_densefs],
                 str(int(time.time()))))

# for fname in [train_types, test_types]:
#    if not os.path.isfile(fname):
#        raise RuntimeError('{} does not exist.'.format(fname))

config_args['data_root'] = data_root
config_args['train'] = train_types
config_args['test'] = test_types
config_args['model'] = ['baseline', 'densefs'][use_densefs]
config_args['batch_size'] = batch_size
config_args['iterations'] = iterations
config_args['save_path'] = savepath

beautify_config(config_args)

gap = 100

e = molgrid.ExampleProvider(data_root=data_root, balanced=True, shuffle=True)
e.populate(train_types)

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(e.num_types())
tensor_shape = (batch_size,) + dims

labels = molgrid.MGrid1f(batch_size)
input_tensor = molgrid.MGrid5f(*tensor_shape)

pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)

losses = []
model = define_densefs_model(
    dims) if use_densefs else define_baseline_model(dims)
model_str = ['Baseline', 'DenseFS'][use_densefs]
print('Using {0} for {1} iterations'.format(model_str, iterations))
plot_model(model, os.path.join(savepath, 'model.png'),
           show_shapes=True)
model.summary()

losses_string = ''
for iteration in range(iterations):
    batch = e.next_batch(batch_size)
    gmaker.forward(batch, input_tensor, 0, random_rotation=True)
    batch.extract_label(0, labels)
    loss = model.train_on_batch(
        input_tensor.tonumpy(), labels.tonumpy())
    losses.append(float(loss))

    losses_string += '{1} loss: {0:0.3f}\n'.format(loss, iteration)
    with open(os.path.join(savepath, 'loss_history_{}.txt'.format(
            model_str.lower())), 'w') as f:
        f.write(losses_string)

    print(iteration, 'loss: {0:0.3f}'.format(loss))

beautify_config(config_args, os.path.join(savepath, 'config'))
model.save(savepath)

losses = [np.mean(losses[window:window + gap])
          for window in np.arange(0, iterations, step=gap)]
plt.plot(np.arange(0, iterations, gap), losses)
plt.legend([model_str])
plt.title('Cross-entropy loss history for DenseFS network'.format(model_str))
plt.savefig(os.path.join(savepath, 'densefs_loss.png'))
print('Finished {}\n\n'.format(model_str))

if test_types is not None:
    e_test = molgrid.ExampleProvider(
        data_root=data_root, balanced=False, shuffle=False)
    e_test.populate(test_types)
    paths, size = get_test_info(e_test, test_types)
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
                s += '{0} | {1:0.3f} {2} {3}\n'.format(
                    int(labels_numpy[i]), predictions[i][1], paths[index][0], paths[index][1])

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
            s += '{0} | {1:0.3f} {2} {3}\n'.format(
                int(labels_numpy[i]), predictions[i][1], paths[index][0], paths[index][1])

    print('Total inference time ({}):'.format(model_str), t.interval, 's')
    with open(os.path.join(savepath, 'predictions_{}.txt'.format(
            model_str.lower())), 'w') as f:
        f.write(s[:-1])
