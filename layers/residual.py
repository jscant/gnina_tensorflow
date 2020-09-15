#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:11:20 2020

@author: scantleb
@brief: Layers involved in building ResNet-based neural networks.

ResNets are founded upon the residual connection; that is, features learned
earlier on in the network are allowed to 'skip' layers, and can be used directly
by deeper layers without going through complex convolutions. This allows the
gradient to flow easier (avoiding the vanishing gradient problem), with the
ultimate goal of easier training of very deep networks.

Paper: https://arxiv.org/abs/1512.03385
"""

import tensorflow as tf
from tensorflow.keras import layers

from layers.layer_functions import generate_activation_layers


class ResBlock(tf.keras.layers.Layer):
    """ResBlock with downsampling and pre-activation."""

    def __init__(self, conv_layers, filters, kernel_size, stride, activation,
                 name):
        """Define block operations.

        Arguments:
            conv_layers: number of convolutional layers in block before final
                convolutional layer
            filters: number of filters in first conv_layers convolutional
                blocks (final conv layer has 4x this number of filters)
            kernel_size: size of convolutional kernels
            stride: factor to decreace spatial dimensions by
            activation: activation function for convolutional layers
            name: name of residual block
        """
        super().__init__()

        conv_initialiser = tf.keras.initializers.HeNormal()
        activations = generate_activation_layers(name, activation)
        bn_axis = 1

        self.conv_layers = []

        for i in range(1, conv_layers + 1):
            ks = 1 if i < conv_layers else kernel_size
            self.conv_layers.append(layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5,
                name=name + '_{}_bn'.format(i)))
            self.conv_layers.append(next(activations))
            self.conv_layers.append(layers.Conv3D(
                4 * filters, ks, 1, name=name + '_{}_conv'.format(i),
                use_bias=False,
                data_format='channels_first', padding='same',
                kernel_initializer=conv_initialiser
            ))

        self.final_bn = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_f_bn')
        self.final_act = next(activations)
        self.final_conv = layers.Conv3D(
            filters, 1, stride, name=name + '_f_conv', use_bias=False,
            data_format='channels_first', padding='same',
            kernel_initializer=conv_initialiser
        )

        self.shortcut_conv_1 = layers.Conv3D(
            filters, 1, stride, name=name + '_sc_conv', use_bias=False,
            data_format='channels_first', padding='same',
            kernel_initializer=conv_initialiser
        )
        self.shortcut_bn_1 = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_sc_bn')

        self.add = layers.Add(name=name + '_add')

    def __call__(self, inputs):
        """Overloaded function; see base class (tf.keras.layers.Layer).

        Returns:
             Tensor containing the result of passing inputs through res block
             defined in the constructor.
        """
        skip_connection = self.shortcut_conv_1(inputs)
        skip_connection = self.shortcut_bn_1(skip_connection)

        x = self.conv_layers[0](inputs)
        for layer in self.conv_layers[1:]:
            x = layer(x)

        x = self.final_bn(x)
        x = self.final_act(x)
        x = self.final_conv(x)

        return self.add([x, skip_connection])


class InverseResBlock(tf.keras.layers.Layer):
    """ResBlock with upsampling and precalculation."""

    def __init__(self, conv_layers, filters, kernel_size, stride, activation,
                 name):
        """Define block operations.

        Arguments:
            conv_layers: number of convolutional layers in block before final
                convolutional layer
            filters: number of filters in first conv_layers convolutional
                blocks (final conv layer has 4x fewer than this number of
                filters)
            kernel_size: size of convolutional kernels
            stride: factor to increase spatial dimensions by
            activation: activation function for convolutional layers
            name: name of residual block
        """
        super().__init__()

        conv_initialiser = tf.keras.initializers.HeNormal()
        activations = generate_activation_layers(name, activation)
        bn_axis = 1

        self.conv_layers = []

        for i in range(1, conv_layers + 1):
            ks = 1 if i < conv_layers else kernel_size
            self.conv_layers.append(layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5,
                name=name + '_{}_bn'.format(i)))
            self.conv_layers.append(next(activations))
            self.conv_layers.append(layers.Conv3DTranspose(
                4 * filters, ks, 1, name=name + '_{}_conv'.format(i),
                use_bias=False,
                data_format='channels_first', padding='same',
                kernel_initializer=conv_initialiser
            ))

        self.final_bn = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_f_bn')
        self.final_act = next(activations)
        self.final_conv = layers.Conv3DTranspose(
            filters, kernel_size, stride, name=name + '_f_conv', use_bias=False,
            data_format='channels_first', padding='same',
            kernel_initializer=conv_initialiser
        )

        self.shortcut_conv_1 = layers.Conv3DTranspose(
            filters, 1, stride, name=name + '_sc_conv', use_bias=False,
            data_format='channels_first', padding='same',
            kernel_initializer=conv_initialiser
        )
        self.shortcut_bn_1 = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_sc_bn')

        self.add = layers.Add(name=name + '_add')

    def __call__(self, inputs):
        """Overloaded function; see base class (tf.keras.layers.Layer).

        Returns:
             Tensor containing the result of passing inputs through res block
             defined in the constructor.
        """
        skip_connection = self.shortcut_conv_1(inputs)
        skip_connection = self.shortcut_bn_1(skip_connection)

        x = self.conv_layers[0](inputs)
        for layer in self.conv_layers[1:]:
            x = layer(x)

        x = self.final_bn(x)
        x = self.final_act(x)
        x = self.final_conv(x)

        return self.add([x, skip_connection])
