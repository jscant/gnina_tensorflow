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
    Reshape, Conv3DTranspose, UpSampling3D
from operator import mul
from functools import reduce


class AutoEncoder(tf.keras.Model):

    def __init__(self, dims, encoding_size=10):
        input_image = Input(shape=dims, dtype=tf.float32, name='input')
        x = Conv3D(16, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(input_image)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        x = Conv3D(32, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        x = Conv3D(32, 3, padding='SAME', activation='relu',
                   data_format='channels_first')(x)
        x = MaxPooling3D(2, 2, data_format='channels_first')(x)
        final_shape = x.shape
        flattened = Flatten(data_format='channels_first')(x)

        encoding = Dense(encoding_size, name='encoding')(flattened)

        decode = Dense(reduce(mul, final_shape[1:]))(encoding)
        reshaped = Reshape(final_shape[1:])(decode)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(reshaped)
        x = Conv3DTranspose(32, 3, padding='SAME', activation='relu',
                            data_format='channels_first')(
            x)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(x)
        x = Conv3DTranspose(16, 3, padding='SAME', activation='relu',
                            data_format='channels_first')(x)
        x = UpSampling3D(
            size=(2, 2, 2), data_format='channels_first')(x)
        reconstruction = Conv3DTranspose(28, 3, padding='SAME', activation='relu',
                                         data_format='channels_first',
                                         name='reconstruction')(x)

        super(AutoEncoder, self).__init__(
            inputs=input_image,
            outputs=[reconstruction, encoding]
        )
        #self.compile(loss=[self.root_relative_squared_error, None],
        #             optimizer=tf.keras.optimizers.SGD())
        self.compile(loss=['mean_absolute_percentage_error', None],
                     optimizer=tf.keras.optimizers.SGD())

    def get_encodings(self, input_tensor):
        _, encoding = self.predict(input_tensor)
        return encoding

    def get_reconstruction(self, input_tensor):
        reconstruction, _ = self.predict(input_tensor)
        return reconstruction

    def root_relative_squared_error(self, target, reconstruction):
        diff = tf.math.squared_difference(target, reconstruction)
        numerator = tf.math.reduce_sum(diff)
        mean = tf.reduce_mean(target)
        dumb_diff = tf.math.squared_difference(target, mean)
        denominator = tf.math.reduce_sum(dumb_diff)
        return tf.sqrt(tf.divide(numerator, denominator))
