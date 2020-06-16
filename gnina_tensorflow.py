# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import argparse
import os
import torch
import molgrid
import numpy as np

from model_definitions import define_baseline_model, define_densefs_model

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("datadir", type=str)
parser.add_argument("typesfile", type=str)
args = parser.parse_args()

molgrid.set_gpu_enabled(False)
data_root = os.path.abspath(args.datadir)
types_fname = os.path.abspath(args.typesfile)

batch_size = 16

e = molgrid.ExampleProvider(data_root=data_root, balanced=True, shuffle=True)
e.populate(types_fname)
batch = e.next_batch(batch_size)

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(e.num_types())
tensor_shape = (batch_size,) + dims

labels = molgrid.MGrid1f(batch_size)
input_tensor = molgrid.MGrid5f(*tensor_shape)


densefs_model = define_densefs_model(dims)
densefs_losses = []
for iteration in range(1000):
    # load data
    batch = e.next_batch(batch_size)

    gmaker.forward(batch, input_tensor, 0, random_rotation=True)
    batch.extract_label(0, labels)

    print(input_tensor.tonumpy().__sizeof__())

    loss = densefs_model.train_on_batch(
        input_tensor.tonumpy(), labels.tonumpy())
    densefs_losses.append(float(loss))
    print(iteration, 'loss: {0:0.3f}'.format(loss))
print('Finished DenseFS\n\n')


baseline_model = define_baseline_model(dims)
baseline_losses = []
for iteration in range(100):
    batch = e.next_batch(batch_size)
    gmaker.forward(batch, input_tensor, 0, random_rotation=True)
    batch.extract_label(1, labels)
    loss = baseline_model.train_on_batch(
        input_tensor.tonumpy(), labels.tonumpy())
    baseline_losses.append(float(loss))
    print(iteration, 'loss: {0:0.3f}'.format(loss))
print('Finished Baseline\n\n')

gap = 20
densefs_losses = [np.mean(densefs_losses[window:window + gap])
                  for window in np.arange(0, 1000, step=gap)]
baseline_losses = [np.mean(baseline_losses[window:window + gap])
                   for window in np.arange(0, 1000, step=gap)]
plt.plot(np.arange(0, 1000, gap), densefs_losses)
plt.plot(np.arange(0, 1000, gap), baseline_losses)
plt.legend(['DenseFS', 'Baseline'])
plt.savefig('losses.png')
