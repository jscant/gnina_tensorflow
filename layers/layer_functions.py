#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:12:15 2020

@author: scantleb
@brief: Functions for generating keras layers.
"""

import tensorflow as tf
from tensorflow.keras import layers


def generate_activation_layers(block_name, activation, n_layers,
                               append_name_info=True):
    """Generate activation layers from strings representing activation layers.

    Arguments:
        block_name: name of the block the layer is a part of
        activation: string representing an activation function; can be
            standard keras string to AF names ('relu', 'sigmoid', etc.), or
            one of either 'prelu' (Parameterised ReLU) or 'threlu'
            (Thresholded ReLU)
        n_layers: number of activation layers to return
        append_name_info: add activation function information to name

    Returns:
        n_layers activation layers with the stipulated activation functions.
    """
    name_template = '{0}_{{0}}_{1}'.format(block_name, activation)

    outputs = []
    for i in range(n_layers):
        if append_name_info:
            act_name = name_template.format(i)
        else:
            act_name = block_name
        if activation == 'prelu':
            outputs.append(
                layers.PReLU(
                    name=act_name,
                    alpha_initializer=tf.keras.initializers.constant(0.1)
                ))
        elif activation == 'threlu':
            outputs.append(
                layers.ThresholdedReLU(theta=1.0, name=act_name)
            )
        else:
            outputs.append(layers.Activation(activation, name=act_name))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)
