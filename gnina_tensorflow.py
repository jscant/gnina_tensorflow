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

data_root = os.path.abspath(args.datadir)
types_fname = os.path.abspath(args.typesfile)

batch_size = 1

e = molgrid.ExampleProvider(data_root=data_root, balanced=True, shuffle=True)
e.populate(types_fname)
batch = e.next_batch(batch_size)

gmaker = molgrid.GridMaker()
dims = gmaker.grid_dimensions(e.num_types())
tensor_shape = (batch_size,) + dims
model = define_densefs_model(dims)

labels = molgrid.MGrid1f(batch_size)
input_tensor = molgrid.MGrid5f(*tensor_shape)

losses = []

# train for 500 iterations
for iteration in range(1000):
    # load data
    batch = e.next_batch(batch_size)

    gmaker.forward(batch, input_tensor, 0, random_rotation=True)
    batch.extract_label(0, labels)

    loss = model.train_on_batch(input_tensor.tonumpy(), labels.tonumpy())
    losses.append(float(loss))
    print(iteration, 'loss: {0:0.3f}'.format(loss))
print('Finished DenseFS\n\n')

model2 = define_baseline_model(dims)
losses_baseline = []
for iteration in range(1000):
    batch = e.next_batch(batch_size)
    gmaker.forward(batch, input_tensor, 0, random_rotation=True)
    batch.extract_label(0, labels)
    loss = model2.train_on_batch(input_tensor.tonumpy(), labels.tonumpy())
    losses_baseline.append(float(loss))
    print(iteration, 'loss: {0:0.3f}'.format(loss))

print('Finished Baseline\n\n')
gap = 20
losses = [np.mean(losses[window:window + gap])
          for window in np.arange(0, 1000, step=gap)]
losses_baseline = [np.mean(losses_baseline[window:window + gap])
                   for window in np.arange(0, 1000, step=gap)]
plt.plot(np.arange(0, 1000, gap), losses)
plt.plot(np.arange(0, 1000, gap), losses_baseline)
plt.legend(['DenseFS', 'Baseline'])
plt.savefig('losses.png')
