#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:46:06 2020

@author: scantleb
@brief: Baseline and DenseFS models.

Baseline (gnina; Ragoza et al., 2017) is the name we have given to the simpler
original CNN built on top of the gnina framework. [1]

DenseFS (Imrie et al., 2018) is a more advanced neural network built on top of
the gnina framework. [2]

DenseNet components modified original tensorflow implementation. [3]

[1] https://pubs.acs.org/doi/10.1021/acs.jcim.8b00350
[2] https://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00740
[3] https://github.com/tensorflow/tensorflow
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend, layers
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense,\
    MaxPooling3D, BatchNormalization, Concatenate, GlobalMaxPooling3D,\
        UpSampling3D, Reshape, Conv3DTranspose
from operator import mul
from functools import reduce


def define_baseline_model(dims):
    """DenseFS network.

    Arguments:
        dims: tuple with input dimensions.
        
    Returns:
        Compiled keras model with DenseFS architecture
    """
    input_layer = Input(shape=dims, dtype=tf.float32)

    # Hidden layers
    x = MaxPooling3D(2, 2, data_format="channels_first",
                     padding='SAME')(input_layer)

    x = Conv3D(filters=32, kernel_size=3, data_format="channels_first",
               activation="relu", padding='SAME')(x)
    x = MaxPooling3D(2, 2, data_format="channels_first", padding='SAME')(x)

    x = Conv3D(filters=64, kernel_size=3, data_format="channels_first",
               activation="relu", padding='SAME')(x)
    x = MaxPooling3D(2, 2, data_format="channels_first", padding='SAME')(x)

    x = Conv3D(filters=128, kernel_size=3, data_format="channels_first",
               activation="relu", padding='SAME')(x)

    # Final layer
    representation = Flatten(data_format="channels_first",
                             name='representation')(x)
    output_layer = Dense(2, activation='softmax')(representation)

    # Compile and return model
    model = keras.models.Model(inputs=input_layer, outputs=[output_layer,
                                                            representation])
    model.compile(optimizer=keras.optimizers.SGD(
        lr=0.01, momentum=0.9), loss=["sparse_categorical_crossentropy",
                                      None])
    # Bug with add_loss puts empty dict at the end of model._layers which
    # interferes with some functionality (such as
    # tf.keras.utils.plot_model)
    model._layers = [layer for layer in model._layers if isinstance(
        layer, tf.keras.layers.Layer)]                                      
    return model


def define_densefs_model(dims, bc=False):
    """DenseFS network.

    Arguments:
        dims: tuple with input dimensions.
        
    Returns:
        Compiled keras model with original gnina architecture
    """
    input_layer = Input(shape=dims, dtype=tf.float32)
    
    if bc:
        db = tf_dense_block
        tb = tf_transition_block
    else:
        db = dense_block
        tb = transition_block

    # Hidden layers
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                     padding='SAME', data_format='channels_first')(input_layer)
    x = Conv3D(32, 3, activation='relu', padding='SAME', use_bias=False,
               data_format='channels_first')(x)

    x = db(x, 4, "db_1")
    x = tb(x, 1.0, "tb_1")

    x = db(x, 4, "db_2")
    x = tb(x, 1.0, "tb_2")

    # Final transition block has global pooling instead of local and no
    # convolution [2]
    x = db(x, 4, "db_3")
    x = tb(x, 1.0, "tb_3", final=True)
    
    representation = GlobalMaxPooling3D(data_format='channels_first',
                                        name='representation')(x)

    # Final layer (already flattened by global max pool)
    output_layer = Dense(2, activation='softmax')(representation)

    # Compile and return model
    model = keras.Model(inputs=input_layer, outputs=[output_layer,
                                                     representation])
    model.compile(optimizer=keras.optimizers.SGD(
        lr=0.01, momentum=0.9), loss=['sparse_categorical_crossentropy', None])
    
    # Bug with add_loss puts empty dict at the end of model._layers which
    # interferes with some functionality (such as
    # tf.keras.utils.plot_model)
    model._layers = [layer for layer in model._layers if isinstance(
        layer, tf.keras.layers.Layer)]
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


def transition_block(x, reduction, name, final=False):
    """A transition block.

    Arguments:
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
        
    Returns:
        output tensor for the block.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_first' else 1
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_bn')(x)

    if final:  # No conv or maxpool, will global pool after final TB
        return x

    x = Conv3D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
               data_format='channels_first', use_bias=False,
               name=name + '_conv', activation='relu')(x)
    x = MaxPooling3D(2, strides=2, name=name + '_pool',
                     data_format='channels_first')(x)
    return x


def inverse_transition_block(x, reduction, name, final=False):
    """A transition block.

    Arguments:
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
        
    Returns:
        output tensor for the block.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_first' else 1
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_bn')(x)

    if final:  # No conv or maxpool, will global pool after final TB
        return x

    x = Conv3D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
               data_format='channels_first', use_bias=False,
               name=name + '_conv', activation='relu')(x)
    x = UpSampling3D(2, name=name + '_upsample',
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

    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5,
        moving_mean_initializer=tf.constant_initializer(0.999),
        name=name + '_1_bn')(x)
    x1 = Conv3D(
        growth_rate, 3, use_bias=False, padding='same',
        activation='relu', name=name + '_1_conv',
        data_format='channels_first')(x1)

    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def tf_dense_block(x, blocks, name):
    """A dense block.
    
    Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.
      
    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = tf_conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def tf_transition_block(x, reduction, name, final=False):
    """A transition block.
    
    Arguments:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.
      
    Returns:
      output tensor for the block.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1
    bn_axis = 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    if final:  # No conv or maxpool, will global pool after final TB
      return x        
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv3D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        name=name + '_conv',
        data_format='channels_first')(x)
    x = layers.MaxPooling3D(2, strides=2, name=name + '_pool',
                                data_format='channels_first')(x)
    return x

def tf_inverse_transition_block(x, reduction, name, final=False):
    """A transition block.
    
    Arguments:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.
      
    Returns:
      output tensor for the block.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1
    bn_axis = 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    if final:  # No conv or maxpool, will global pool after final TB
      return x        
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv3D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        name=name + '_conv',
        data_format='channels_first')(x)
    x = layers.UpSampling3D(2, name=name + '_upsampling', data_format='channels_first')(x)
    return x


def tf_conv_block(x, growth_rate, name):
    """A building block for a dense block.
    
    Arguments:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.
      
    Returns:
      Output tensor for the block.
    """
    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1
    bn_axis = 1
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
            x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv3D(
        4 * growth_rate, 1, use_bias=False, name=name + '_1_conv',
        data_format='channels_first')(x1)
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
            x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv3D(
        growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv',
        data_format='channels_first')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x