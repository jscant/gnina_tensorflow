#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:46:06 2020

@author: scantleb
@brief: Baseline and DenseFS models. Densenet components modified from original
tensorflow implementation found at
https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, MaxPooling3D, BatchNormalization, Concatenate


def define_baseline_model(dims):
    """DenseFS network.
    Arguments:
        dims: tuple with input dimensions.
    Returns:
        Compiled keras model with DenseFS architecture
    """
    input_layer = Input(shape=dims, dtype=tf.float32)

    # Hidden layers
    x = MaxPooling3D(2, 2, data_format="channels_first", padding='SAME')(input_layer)
    x = Conv3D(filters=32, kernel_size=3,
               data_format="channels_first", activation="relu", padding='SAME')(x)
    x = MaxPooling3D(2, 2, data_format="channels_first", padding='SAME')(x)
    x = Conv3D(filters=64, kernel_size=3,
               data_format="channels_first", activation="relu", padding='SAME')(x)
    x = MaxPooling3D(2, 2, data_format="channels_first", padding='SAME')(x)
    x = Conv3D(filters=128, kernel_size=3,
               data_format="channels_first", activation="relu", padding='SAME')(x)

    # Final layer
    x = Flatten(data_format="channels_first")(x)
    output_layer = Dense(2, activation='softmax')(x)

    # Compile and return model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.SGD(
        lr=0.01, momentum=0.9), loss="sparse_categorical_crossentropy")

    return model

def _define_baseline_model(dims):
    input_layer = keras.layers.Input(shape=dims)
    pool0 = keras.layers.MaxPooling3D(data_format="channels_first")(input_layer)
    conv1 = keras.layers.Conv3D(filters=32, kernel_size=3, data_format="channels_first", activation="relu")(pool0)
    pool1 = keras.layers.MaxPooling3D(data_format="channels_first")(conv1)
    conv2 = keras.layers.Conv3D(filters=64, kernel_size=3, data_format="channels_first", activation="relu")(pool1)
    pool2 = keras.layers.MaxPooling3D(data_format="channels_first")(conv2)
    conv3 = keras.layers.Conv3D(filters=128, kernel_size=3, data_format="channels_first", activation="relu")(pool2)

    flatten = keras.layers.Flatten(data_format="channels_first")(conv3)

    fc1 = keras.layers.Dense(2,activation='softmax')(flatten)

    # Define and return model
    model = keras.models.Model(inputs=input_layer, outputs=fc1)
    model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9), loss="sparse_categorical_crossentropy")

    return model


def define_densefs_model(dims):
    """DenseFS network.
    Arguments:
        dims: tuple with input dimensions.
    Returns:
        Compiled keras model with original gnina architecture
    """
    input_layer = Input(shape=dims, dtype=tf.float32)

    # Hidden layers
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                     padding='SAME', data_format='channels_first')(input_layer)
    x = Conv3D(32, 3, activation='relu', padding='SAME',
               data_format='channels_first')(x)
    x = dense_block(x, 4, "db_1")
    x = transition_block(x, 1.0, "tb_1")
    x = dense_block(x, 4, "db_2")
    x = transition_block(x, 1.0, "tb_2")
    x = dense_block(x, 4, "db_3")
    x = transition_block(x, 1.0, "tb_3")

    # Final layer
    x = Flatten(data_format='channels_first')(x)
    output_layer = Dense(2, activation='softmax')(x)

    # Compile and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adadelta(
        lr=0.01), loss='sparse_categorical_crossentropy')
    return model


def dense_block(x, blocks, name):
    """A dense block.

    Arguments:
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    Returns:
        Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 16, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.

    Arguments:
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    Returns:
        output tensor for the block.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_first' else 1
    x = Conv3D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        data_format='channels_first',
        use_bias=False,
        name=name + '_conv', activation='relu')(
        x)
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(
        x)
    x = MaxPooling3D(2, strides=2, name=name + '_pool',
                     data_format='channels_first')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    Arguments:
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    Returns:
        Output tensor for the block.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_first' else 1

    x1 = Conv3D(
        growth_rate, 1, use_bias=False, padding='same',
        activation='relu', name=name + '_1_conv',
        data_format='channels_first')(x)
    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
        x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x
