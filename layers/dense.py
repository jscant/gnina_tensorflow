#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 14:26:40 2020
@author: scantleb
@brief: Layer definitions for dense-net based networks.

Includes both the 'old-style' dense blocks as well as the updated DenseNet-BC
version, as well as 'inverse-' dense and transition blocks, for upsampling in
autoencoder.
"""

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

from layers.layer_functions import generate_activation_layers


def dense_block(x, blocks, name, activation='relu'):
    """A dense block.

    Arguments:
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
        activation: keras activation to be applied for each conv block

    Returns:
        Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(
            x, 16, name=name + '_block' + str(i + 1), activation=activation)
    return x


def transition_block(x, reduction, name, activation='relu', final=False):
    """A transition block.

    Arguments:
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
        activation: activation function for convolution.
        final: if True, only apply batch normalisation.

    Returns:
        output tensor for the block.
    """
    conv_initialiser = tf.keras.initializers.HeNormal()
    act_0 = next(generate_activation_layers(name, activation))
    bn_axis = 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_bn')(x)

    if final:  # No conv or maxpool, will global pool after final TB
        return x

    x = layers.Conv3D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      data_format='channels_first', use_bias=False,
                      kernel_initializer=conv_initialiser,
                      name=name + '_{}'.format(activation))(x)
    x = act_0(x)
    x = layers.MaxPooling3D(2, strides=2, name=name + '_pool',
                            data_format='channels_first')(x)
    return x


def inverse_transition_block(x, reduction, name, activation='relu'):
    """A transition block.

    Arguments:
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
        activation: activation function for convolution.

    Returns:
        output tensor for the block.
    """
    conv_initialiser = tf.keras.initializers.HeNormal()
    act_0 = next(generate_activation_layers(name, activation))
    bn_axis = 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_bn')(x)

    x = layers.Conv3D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      data_format='channels_first', use_bias=False,
                      kernel_initializer=conv_initialiser,
                      name=name + '_{}'.format(activation))(x)
    x = act_0(x)
    x = layers.UpSampling3D(2, name=name + '_upsample',
                            data_format='channels_first')(x)
    return x


def conv_block(x, growth_rate, name, activation='relu'):
    """A building block for a dense block.

    Arguments:
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
        activation: activation function for convolution.

    Returns:
        Output tensor for the block.
    """
    conv_initialiser = tf.keras.initializers.HeNormal()
    act_0 = next(generate_activation_layers(name, activation))
    bn_axis = 1
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_0_bn')(x)
    x1 = layers.Conv3D(
        growth_rate, 3, use_bias=False, padding='same',
        name=name + '_0_{}'.format(activation),
        kernel_initiaer=conv_initialiser,
        data_format='channels_first')(x1)
    x1 = act_0(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def tf_dense_block(x, blocks, name, activation='relu'):
    """A dense block.

    Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.
      activation: activation function for convolution.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = tf_conv_block(
            x, 32, name=name + '_block' + str(i + 1), activation=activation)
    return x


def tf_transition_block(x, reduction, name, activation='relu', final=False):
    """A transition block.

    Arguments:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.
      activation: activation function for convolution.
      final: if True, only apply batch normalisation.

    Returns:
      output tensor for the block.
    """
    conv_initialiser = tf.keras.initializers.HeNormal()
    act_0 = next(generate_activation_layers(name, activation))
    bn_axis = 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    if final:  # No conv or maxpool, will global pool after final TB
        return x
    x = act_0(x)
    x = layers.Conv3D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        kernel_initializer=conv_initialiser,
        name=name + '_conv',
        data_format='channels_first')(x)
    x = layers.MaxPooling3D(2, strides=2, name=name + '_pool',
                            data_format='channels_first')(x)
    return x


def tf_inverse_transition_block(x, reduction, name, activation='relu'):
    """A transition block.

    Arguments:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.
      activation: activation function for convolution.

    Returns:
      output tensor for the block.
    """
    conv_initialiser = tf.keras.initializers.HeNormal()
    act_0 = next(generate_activation_layers(name, activation))
    bn_axis = 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = act_0(x)
    x = layers.Conv3D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        kernel_initializer=conv_initialiser,
        name=name + '_conv',
        data_format='channels_first')(x)
    x = layers.UpSampling3D(2, name=name + '_upsampling',
                            data_format='channels_first')(x)
    return x


def tf_conv_block(x, growth_rate, name, activation='relu'):
    """A building block for a dense block.

    Arguments:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.
      activation: activation function for convolution.

    Returns:
      Output tensor for the block.
    """
    conv_initialiser = tf.keras.initializers.HeNormal()
    activations = generate_activation_layers(name, activation)
    bn_axis = 1
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
        x)
    x1 = next(activations)(x1)
    x1 = layers.Conv3D(
        4 * growth_rate, 1, use_bias=False, name=name + '_1_conv',
        data_format='channels_first',
        kernel_initializer=conv_initialiser)(x1)
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
        x1)
    x1 = next(activations)(x1)
    x1 = layers.Conv3D(
        growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv',
        data_format='channels_first', kernel_initializer=conv_initialiser)(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x
