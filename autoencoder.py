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
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, MaxPooling3D,\
    Reshape, Conv3DTranspose, UpSampling3D, BatchNormalization
from operator import mul
from functools import reduce


class AutoEncoder(tf.keras.Model):

    def __init__(self, dims, encoding_size=10):
        """Setup for autoencoder architecture."""
        input_image = Input(shape=dims, dtype=tf.float32, name='input')
        x = Conv3D(16, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(input_image)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        x = Conv3D(32, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        x = Conv3D(32, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        final_shape = x.shape
        flattened = Flatten(data_format='channels_first')(x)

        encoding = Dense(encoding_size, name='encoding')(flattened)

        decode = Dense(reduce(mul, final_shape[1:]))(encoding)
        reshaped = Reshape(final_shape[1:])(decode)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(reshaped)
        x = Conv3DTranspose(32, 3, padding='SAME', activation='relu',
                            data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(x)
        x = Conv3DTranspose(16, 3, padding='SAME', activation='relu',
                            data_format='channels_first')(x)
        x = BatchNormalization(
            axis=4, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999))(x)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(x)
        reconstruction = Conv3DTranspose(28, 3, padding='SAME',
                                         activation='relu',
                                         data_format='channels_first',
                                         name='reconstruction')(x)
        
        frac = Input(shape=(1,), dtype=tf.float32)

        super(AutoEncoder, self).__init__(
            inputs=[input_image, frac],
            outputs=[reconstruction, encoding]
        )
        self.add_loss(self.composite_mse(input_image, reconstruction, frac))
        self.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9),
            metrics={'reconstruction': [self.zero_mse, self.nonzero_mse]}
        )

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
        frac = ratio/(1+ratio)
        return frac*self.nonzero_mse(target, reconstruction) + (1-frac)*self.zero_mse(target, reconstruction)
