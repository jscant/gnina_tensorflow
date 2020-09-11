#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 01 14:26:40 2020
@author: scantleb
@brief: Layer definitions for autoencoder and classifier. Includes both the
        'old-style' dense blocks as well as the updated DenseNet-BC version,
        as well as 'inverse-' dense and transition blocks, for upsampling in
        autoencoder.
"""

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, \
    UpSampling3D, Concatenate, Activation, PReLU, ThresholdedReLU, Add, \
    Conv3DTranspose


class ResBlock(tf.keras.layers.Layer):

    def __init__(self, conv_layers, filters, kernel_size, stride, activation,
                 name):
        super().__init__()

        conv_initialiser = tf.keras.initializers.HeNormal()
        activations = generate_activation_layers(
            name, activation, conv_layers + 1)
        bn_axis = 1

        self.conv_layers = []

        for i in range(1, conv_layers + 1):
            ks = 1 if i < conv_layers else kernel_size
            self.conv_layers.append(BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5,
                name=name + '_{}_bn'.format(i)))
            self.conv_layers.append(activations[i - 1])
            self.conv_layers.append(Conv3D(
                filters, ks, 1, name=name + '_{}_conv'.format(i),
                use_bias=False,
                data_format='channels_first', padding='same',
                kernel_initializer=conv_initialiser
            ))

        self.final_bn = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_f_bn')
        self.final_act = activations[-1]
        self.final_conv = Conv3D(
            4 * filters, 1, stride, name=name + '_f_conv', use_bias=False,
            data_format='channels_first', padding='same',
            kernel_initializer=conv_initialiser
        )

        self.shortcut_conv_1 = Conv3D(
            4 * filters, 1, stride, name=name + '_sc_conv', use_bias=False,
            data_format='channels_first', padding='same',
            kernel_initializer=conv_initialiser
        )
        self.shortcut_bn_1 = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_sc_bn')

        self.add = Add(name=name + '_add')

    def __call__(self, inputs):
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

    def __init__(self, conv_layers, filters, kernel_size, stride, activation,
                 name):
        super().__init__()

        conv_initialiser = tf.keras.initializers.HeNormal()
        activations = generate_activation_layers(
            name, activation, conv_layers + 1)
        bn_axis = 1

        self.conv_layers = []

        for i in range(1, conv_layers + 1):
            ks = 1 if i < conv_layers else kernel_size
            self.conv_layers.append(BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5,
                name=name + '_{}_bn'.format(i)))
            self.conv_layers.append(activations[i - 1])
            self.conv_layers.append(Conv3DTranspose(
                filters, ks, 1, name=name + '_{}_conv'.format(i),
                use_bias=False,
                data_format='channels_first', padding='same',
                kernel_initializer=conv_initialiser
            ))

        self.final_bn = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_f_bn')
        self.final_act = activations[-1]
        self.final_conv = Conv3DTranspose(
            filters // 4, 1, stride, name=name + '_f_conv', use_bias=False,
            data_format='channels_first', padding='same',
            kernel_initializer=conv_initialiser
        )

        self.shortcut_conv_1 = Conv3DTranspose(
            filters // 4, 1, stride, name=name + '_sc_conv', use_bias=False,
            data_format='channels_first', padding='same',
            kernel_initializer=conv_initialiser
        )
        self.shortcut_bn_1 = BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_sc_bn')

        self.add = Add(name=name + '_add')

    def __call__(self, inputs):
        skip_connection = self.shortcut_conv_1(inputs)
        skip_connection = self.shortcut_bn_1(skip_connection)

        x = self.conv_layers[0](inputs)
        for layer in self.conv_layers[1:]:
            x = layer(x)

        x = self.final_bn(x)
        x = self.final_act(x)
        x = self.final_conv(x)

        return self.add([x, skip_connection])


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
                PReLU(
                    name=act_name,
                    alpha_initializer=tf.keras.initializers.constant(0.1)
                ))
        elif activation == 'threlu':
            outputs.append(
                ThresholdedReLU(theta=1.0, name=act_name)
            )
        else:
            outputs.append(Activation(activation, name=act_name))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


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
    act_0 = generate_activation_layers(name, activation, 1)
    bn_axis = 1
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_bn')(x)

    if final:  # No conv or maxpool, will global pool after final TB
        return x

    x = Conv3D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
               data_format='channels_first', use_bias=False,
               kernel_initializer=conv_initialiser,
               name=name + '_{}'.format(activation))(x)
    x = act_0(x)
    x = MaxPooling3D(2, strides=2, name=name + '_pool',
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
    act_0 = generate_activation_layers(name, activation, 1)
    bn_axis = 1
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_bn')(x)

    x = Conv3D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
               data_format='channels_first', use_bias=False,
               kernel_initializer=conv_initialiser,
               name=name + '_{}'.format(activation))(x)
    x = act_0(x)
    x = UpSampling3D(2, name=name + '_upsample',
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
    act_0, _ = generate_activation_layers(name, activation, 1)
    bn_axis = 1
    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_0_bn')(x)
    x1 = Conv3D(
        growth_rate, 3, use_bias=False, padding='same',
        name=name + '_0_{}'.format(activation),
        kernel_initiaer=conv_initialiser,
        data_format='channels_first')(x1)
    x1 = act_0(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
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
    act_0, _ = generate_activation_layers(name, activation, 1)
    bn_axis = 1
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    if final:  # No conv or maxpool, will global pool after final TB
        return x
    x = act_0(x)
    x = Conv3D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        kernel_initializer=conv_initialiser,
        name=name + '_conv',
        data_format='channels_first')(x)
    x = MaxPooling3D(2, strides=2, name=name + '_pool',
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
    act_0, _ = generate_activation_layers(name, activation, 1)
    bn_axis = 1
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = act_0(x)
    x = Conv3D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        kernel_initializer=conv_initialiser,
        name=name + '_conv',
        data_format='channels_first')(x)
    x = UpSampling3D(2, name=name + '_upsampling',
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
    act_0, act_1 = generate_activation_layers(name, activation, 1)
    bn_axis = 1
    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
        x)
    x1 = act_0(x1)
    x1 = Conv3D(
        4 * growth_rate, 1, use_bias=False, name=name + '_1_conv',
        data_format='channels_first',
        kernel_initializer=conv_initialiser)(x1)
    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
        x1)
    x1 = act_1(x1)
    x1 = Conv3D(
        growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv',
        data_format='channels_first', kernel_initializer=conv_initialiser)(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x
