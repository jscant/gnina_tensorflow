#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:30:08 2020

@author: scantleb
@brief: AutoEncoder class definition

Autoencoders learn a mapping from a high dimensional space to a lower
dimensional space, as well as the inverse.
"""


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, \
    MaxPooling3D, Reshape, Conv3DTranspose, UpSampling3D, BatchNormalization, \
        GlobalMaxPooling3D
from operator import mul
from functools import reduce
from model_definitions import tf_transition_block,\
    tf_inverse_transition_block, tf_dense_block
import os


class AutoEncoderBase(tf.keras.Model):
    """Virtual parent class for autoencoders."""
    
    def __init__(self, optimiser, lr, momentum=0.0):
        """Initialisation of base class.
        
        Arguments:
            optimiser: any keras optimisation class
            lr: learning rate of the optimiser
            momentum: momentum for the optimiser (where applicable)
        """
        super(AutoEncoderBase, self).__init__(
            inputs=[self.input_image, self.frac],
            outputs=[self.reconstruction, self.encoding]
        )

        self.add_loss(self.composite_mse(
            self.input_image, self.reconstruction, self.frac))
        #self.add_loss(tf.keras.losses.mse(self.input, self.reconstruction))
        try:
            self.compile(
                optimizer=optimiser(lr=lr, momentum=momentum),
                metrics={'reconstruction': [self.zero_mse, self.nonzero_mse]}
            )
        except TypeError:
            self.compile(
                optimizer=optimiser(lr=lr),
                metrics={'reconstruction': [self.zero_mse, self.nonzero_mse]}
            )
            
        # Bug with add_loss puts empty dict at the end of model._layers which
        # interferes with some functionality (such as
        # tf.keras.utils.plot_model)
        self._layers = [layer for layer in self._layers if isinstance(
            layer, tf.keras.layers.Layer)]
        
    def long_sigmoid(self, x):
        """2.3 times the sigmoid function."""
        return tf.math.multiply(2.3, tf.nn.sigmoid(x))

    def nonzero_mse(self, target, reconstruction):
        """Mean squared error for non-zero values in the target matrix
        
        Finds the mean squared error for all parts of the input tensor which
        are not equal to zero.
        
        Arguments:
            target: input tensor
            reconstruction: output tensor of the autoencoder
            
        Returns:
            Mean squared error for all non-zero entries in the target
        """
        mask = 1. - tf.cast(tf.equal(target, 0), tf.float32)
        masked_difference = (target - reconstruction) * mask
        squared_difference = tf.reduce_sum(tf.square(masked_difference))
        divided_sd = tf.divide(squared_difference, tf.reduce_sum(mask))
        return tf.sqrt(divided_sd)
    
    def zero_mse(self, target, reconstruction):
        """Mean squared error for zero values in the target matrix
        
        Finds the mean squared error for all parts of the input tensor which
        are equal to zero.
        
        Arguments:
            target: input tensor
            reconstruction: output tensor of the autoencoder
            
        Returns:
            Mean squared error for all zero entries in the target
        """
        mask = tf.cast(tf.equal(target, 0), tf.float32)
        masked_difference = (target - reconstruction) * mask
        squared_difference = tf.reduce_sum(tf.square(masked_difference))
        divided_sd = tf.divide(squared_difference, tf.reduce_sum(mask))
        return tf.sqrt(divided_sd)
    
    def composite_mse(self, target, reconstruction, ratio):
        """Weighted mean squared error of nonzero-only and zero-only inputs.
        
        Finds the MSE between the autoencoder reconstruction and the nonzero
        entries of the input, the MSE between the reconstruction and the zero
        entries of the input and gives the weighted average of the two.
        
        Arguments:
            target: input tensor
            reconstruction: output tensor of the autoencoder
            ratio: desired ratio of nonzero : zero
            
        Returns:
            Average weighted by:
                
                ratio/(1+ratio)*nonzero_mse + 1/(1+ratio)*zero_mse
                
            where nonzero_mse and zero_mse are the MSE for the nonzero and zero
            parts of target respectively.
        """
        #frac = tf.cast(tf.math.divide(ratio, tf.math.add(1, ratio)), tf.float32)
        frac = ratio/(1+ratio)
        return tf.math.add(
            tf.math.multiply(frac, self.nonzero_mse(target, reconstruction)),
            tf.math.multiply(1-frac, self.zero_mse(target, reconstruction)))

class AutoEncoder(AutoEncoderBase):

    def __init__(self, dims, encoding_size=10,
                 optimiser=tf.keras.optimizers.SGD, lr=0.01, momentum=0.0):
        """Setup for autoencoder architecture."""
        self.optimiser = optimiser
        self.lr = lr
        self.momentum = momentum
        self.dims = dims
        self.encoding_size = encoding_size
        
        self.input_image = Input(shape=dims, dtype=tf.float32, name='input')
        x = Conv3D(32, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(self.input_image)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        x = Conv3D(64, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        x = Conv3D(128, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        x = Conv3D(256, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = Conv3D(256, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        
        final_shape = x.shape
        flattened = Flatten(data_format='channels_first')(x)

        self.encoding = Dense(
            encoding_size, name='encoding', activation='sigmoid')(flattened)

        decode = Dense(reduce(mul, final_shape[1:]))(self.encoding)
        reshaped = Reshape(final_shape[1:])(decode)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(reshaped)
        x = Conv3DTranspose(128, 3, padding='SAME', activation='relu',
                            data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(x)
        x = Conv3DTranspose(64, 3, padding='SAME', activation='relu',
                            data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(x)
        x = Conv3DTranspose(32, 3, padding='SAME', activation='relu',
                            data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(x)
        self.reconstruction = Conv3DTranspose(28, 3, padding='SAME',
                                         activation='linear',
                                         data_format='channels_first',
                                         name='reconstruction')(x)
        self.frac = Input(shape=(1,), dtype=tf.float32)

        super(AutoEncoder, self).__init__(
            optimiser, lr, momentum
        )


class DenseAutoEncoder(AutoEncoderBase):
    """Densely connected autoencoder."""
    
    def __init__(self, dims, encoding_size=10,
                 optimiser=tf.keras.optimizers.SGD, lr=0.01, momentum=0.0):
        """Setup for autoencoder architecture."""
        self.optimiser = optimiser
        self.lr = lr
        self.momentum = momentum
        self.dims = dims
        self.encoding_size = encoding_size
        
        self.input_image = Input(shape=dims, dtype=tf.float32)
    
        # Hidden layers
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='SAME', data_format='channels_first')(
                             self.input_image)
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
        #x = tf_dense_block(x, 4, "db_4")
        #x = tf_transition_block(x, 0.5, "tb_4", final=True)
        
        final_shape = x.shape
        x = Flatten(data_format='channels_first')(x)
        #x = GlobalMaxPooling3D(data_format='channels_first')(x)
        self.encoding = Dense(encoding_size, activation='sigmoid',
                         name='encoding')(x)
        
        decoding = Dense(reduce(mul, final_shape[1:]))(self.encoding)
        reshaped = Reshape(final_shape[1:])(decoding)
        
        #x = Conv3D(32, 2, activation='relu', padding='SAME', use_bias=False,
        #           data_format='channels_first')(reshaped)
        
        x = tf_inverse_transition_block(reshaped, 0.5, 'itb_1')
        x = tf_dense_block(x, 4, 'idb_1')
        
        x = tf_inverse_transition_block(x, 0.5, 'itb_2')
        x = tf_dense_block(x, 4, 'idb_2')
        
        x = tf_inverse_transition_block(x, 0.5, 'itb_3')
        x = tf_dense_block(x, 4, 'idb_3')
         
        x = tf_inverse_transition_block(x, 0.5, 'itb_4')
        x = tf_dense_block(x, 4, 'idb_4')
    
        self.reconstruction = Conv3D(28, 3, activation=self.long_sigmoid,
                                data_format='channels_first', padding='SAME',
                                name='reconstruction')(x)
        
        self.frac = Input(shape=(1,), dtype=tf.float32)

        super(DenseAutoEncoder, self).__init__(
            optimiser, lr, momentum
        )    


if __name__ == '__main__':
    from contextlib import redirect_stdout
    tf.keras.backend.clear_session()
    d = DenseAutoEncoder((28, 48, 48, 48), encoding_size=100)
    
    with open('file_2.model', 'w') as f:
        with redirect_stdout(f):
            d.summary()