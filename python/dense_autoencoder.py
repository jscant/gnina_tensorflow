#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:29:39 2020

@author: scantleb
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend, layers
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense,\
    MaxPooling3D, BatchNormalization, Concatenate, GlobalMaxPooling3D,\
        UpSampling3D, Reshape, Conv3DTranspose
from operator import mul
from functools import reduce

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
    x = layers.AveragePooling3D(2, strides=2, name=name + '_pool',
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
    x = Conv3D(32, 3, activation='relu', padding='SAME', use_bias=False,
               data_format='channels_first')(x)

    x = tf_dense_block(x, 4, "db_1")
    x = tf_transition_block(x, 0.5, "tb_1")

    x = tf_dense_block(x, 4, "db_2")
    x = tf_transition_block(x, 0.5, "tb_2")

    # Final transition block has global pooling instead of local and no
    # convolution [2]
    x = tf_dense_block(x, 4, "db_3")
    x = tf_transition_block(x, 0.5, "tb_3", final=False)
    x = tf_dense_block(x, 4, "db_4")
    x = tf_transition_block(x, 0.5, "tb_4", final=True)
    final_shape = x.shape
    x = GlobalMaxPooling3D(data_format='channels_first')(x)
    
    encoding = Dense(1000, activation='relu', name='encoding')(x)
    
    decoding = Dense(reduce(mul, final_shape[1:]))(encoding)
    reshaped = Reshape(final_shape[1:])(decoding)
    
    x = Conv3D(32, 2, activation='relu', padding='SAME', use_bias=False,
               data_format='channels_first')(reshaped)
    x = tf_inverse_transition_block(x, 0.5, 'itb_1')
    x = tf_dense_block(x, 4, 'idb_1')
    
    x = tf_inverse_transition_block(x, 0.5, 'itb_2')
    x = tf_dense_block(x, 4, 'idb_2')
    
    x = tf_inverse_transition_block(x, 0.5, 'itb_3')
    x = tf_dense_block(x, 4, 'idb_3')
     
    #x = UpSampling3D(2, data_format='channels_first')(x)
    x = tf_inverse_transition_block(x, 0.5, 'itb_4')
    x = tf_dense_block(x, 4, 'idb_4')
    output_layer = Conv3D(28, 3, activation='linear', data_format='channels_first',
               padding='SAME')(x)
    
    
    # Compile and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.SGD(
        lr=0.01, momentum=0.9), loss='sparse_categorical_crossentropy')
    return model

if __name__ == '__main__':
    define_densefs_model((28, 48, 48, 48)).summary()