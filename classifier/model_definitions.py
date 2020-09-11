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
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, \
    MaxPooling3D, GlobalMaxPooling3D

from layers import dense


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
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
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
        bc: use DenseNet-BC (updated DenseNet architecture)

    Returns:
        Compiled keras model with original gnina architecture
    """
    input_layer = Input(shape=dims, dtype=tf.float32)

    if bc:
        db = dense.tf_dense_block
        tb = dense.tf_transition_block
    else:
        db = dense.dense_block
        tb = dense.transition_block

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
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.SGD(
        lr=0.01, momentum=0.9), loss=['sparse_categorical_crossentropy', None])

    # Bug with add_loss puts empty dict at the end of model._layers which
    # interferes with some functionality (such as
    # tf.keras.utils.plot_model)
    model._layers = [layer for layer in model._layers if isinstance(
        layer, tf.keras.layers.Layer)]
    return model
