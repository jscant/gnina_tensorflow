"""
Created on Sat Jun 20 12:30:08 2020

@author: scantleb
@brief: AutoEncoder class definition

Autoencoders learn a mapping from a high dimensional space to a lower
dimensional space, as well as the inverse.
"""


import tensorflow as tf
import numpy as np

from classifier.model_definitions import tf_transition_block,\
    tf_inverse_transition_block, tf_dense_block
from operator import mul
from functools import reduce
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, \
    MaxPooling3D, Reshape, Conv3DTranspose, UpSampling3D, BatchNormalization


class AutoEncoderBase(tf.keras.Model):
    """Virtual parent class for autoencoders."""

    def __init__(self, optimiser, loss, opt_args):
        """Initialisation of base class.

        Arguments:
            optimiser: any keras optimisation class
            lr: learning rate of the optimiser
            momentum: momentum for the optimiser (where applicable)
        """
        
        optimisers = {
            'sgd': tf.keras.optimizers.SGD,
            'adadelta': tf.keras.optimizers.Adadelta,
            'adagrad': tf.keras.optimizers.Adagrad,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adamax': tf.keras.optimizers.Adamax,
            'adam': tf.keras.optimizers.Adam
        }
        
        optimiser = optimisers.get(optimiser, tf.keras.optimizers.SGD)

        inputs = [self.input_image]
        if loss == 'composite_mse':
            self.frac = Input(shape=(1,), dtype=tf.float32, name='frac')
            inputs.append(self.frac)

        super(AutoEncoderBase, self).__init__(
            inputs=inputs,
            outputs=[self.reconstruction, self.encoding]
        )

        metrics = {'reconstruction': [self.mae, self.nonzero_mae, self.zero_mae,
                                      self.zero_mse, self.nonzero_mse]}
        if loss == 'composite_mse':
            self.add_loss(self.composite_mse(
                self.input_image, self.reconstruction, self.frac))
            self.compile(
                optimizer=optimiser(**opt_args),
                metrics=metrics
            )
        else:
            if loss == 'unbalanced':
                loss = self.unbalanced_loss
            self.compile(
                optimizer=optimiser(**opt_args),
                loss={'reconstruction': loss,
                      'encoding': None},
                metrics=metrics
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
        return tf.reduce_mean(tf.square(masked_difference))

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
        return tf.reduce_mean(tf.square(masked_difference))

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
        frac = tf.divide(ratio, 1+ratio)
        return tf.math.add(
            tf.math.multiply(frac, self.nonzero_mse(target, reconstruction)),
            tf.math.multiply(1-frac, self.zero_mse(target, reconstruction)))

    def mae(self, target, reconstruction):
        """Mean absolute error loss function.

        Arguments:
            target: input tensor
            reconstruction: output tensor of the autoencoder

        Returns:
            Tensor containing the mean absolute error between the target and
            the reconstruction.
        """
        return tf.reduce_mean(tf.abs(target - reconstruction))

    def zero_mae(self, target, reconstruction):
        """Mean absolute error loss function target values are zero.

        Arguments:
            target: input tensor
            reconstruction: output tensor of the autoencoder

        Returns:
            Tensor containing the mean absolute error between the target and
            the reconstruction where the mean is taken over values where
            the target is equal to zero.
            This can be NaN if there are no inputs equal to zero.
        """
        mask = tf.cast(tf.equal(target, 0), tf.float32)
        masked_diff = (target - reconstruction) * mask
        abs_diff = tf.abs(masked_diff)
        return tf.divide(tf.reduce_sum(abs_diff), tf.reduce_sum(mask))

    def nonzero_mae(self, target, reconstruction):
        """Mean absolute error loss function target values are not zero.

        Arguments:
            target: input tensor
            reconstruction: output tensor of the autoencoder

        Returns:
            Tensor containing the mean absolute error between the target and
            the reconstruction where the mean is taken over values where
            the target is not zero.
            This can be NaN if there are no nonzero inputs.
        """
        mask = 1 - tf.cast(tf.equal(target, 0), tf.float32)
        mask_sum = tf.reduce_sum(mask)
        masked_diff = (target - reconstruction) * mask
        abs_diff = tf.abs(masked_diff)
        return tf.divide(tf.reduce_sum(abs_diff), mask_sum)

    def unbalanced_loss(self, y_true, y_pred):
        """Loss function which more heavily penalises loss on nonzero terms.

        Only use for positive labels.

        Arguments:
            y_true: tensor containing true labels for model inputs
            y_pred: tensor containing output(s) of model

        Returns:
            Error weighted strongly in favour of penalising errors on parts
            of y_true which are above zero.
        """
        term_1 = tf.pow(y_true - y_pred, 4)
        term_2 = tf.pow(y_true - y_pred, 2)
        loss = y_true*term_1 + tf.multiply(
            tf.constant(0.01), tf.multiply(1 - y_true, term_2))
        return tf.reduce_mean(loss)

    def approx_heaviside(self, x):
        """Activation function: continuous approximation to Heaviside fn.

        Arguments:
            x: Tensor with pre-activation values.

        Returns:
            Tensor containing activations, where activations at or below zero
            are (approx.) zero, and activations above zero are (approx) one.
        """
        return 0.5 + 0.5*tf.tanh(x*10.)

    def _define_activations(self):
        """Returns dict from strings to final layer activations objects."""
        
        return {'sigmoid': 'sigmoid',
                'relu': 'relu',
                'heaviside': self.approx_heaviside}


class AutoEncoder(AutoEncoderBase):

    def __init__(self,
                 dims,
                 encoding_size=10,
                 optimiser=tf.keras.optimizers.SGD,
                 loss='mse',
                 final_activation='sigmoid',
                 **opt_args):
        """Setup for autoencoder architecture."""

        activations = super(AutoEncoder, self)._define_activations()
        activation = activations.get(final_activation, 'linear')

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

        self.reconstruction = Conv3DTranspose(
            dims[0], 3, padding='SAME',
            activation=activation,
            data_format='channels_first', name='reconstruction')(x)

        super(AutoEncoder, self).__init__(
            optimiser, loss, opt_args
        )


class DenseAutoEncoder(AutoEncoderBase):
    """Autoencoder with Dense connectivity."""

    def __init__(self, dims, encoding_size=10,
                 optimiser=tf.keras.optimizers.SGD,
                 loss='mse',
                 final_activation='sigmoid',
                 **opt_args):
        """Setup for autoencoder architecture."""

        activations = super(DenseAutoEncoder, self)._define_activations()
        activation = activations.get(final_activation, 'linear')

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

        final_shape = x.shape
        x = Flatten(data_format='channels_first')(x)

        self.encoding = Dense(encoding_size, activation='sigmoid',
                              name='encoding')(x)

        decoding = Dense(reduce(mul, final_shape[1:]))(self.encoding)
        reshaped = Reshape(final_shape[1:])(decoding)

        x = tf_inverse_transition_block(reshaped, 0.5, 'itb_1')
        x = tf_dense_block(x, 4, 'idb_1')

        x = tf_inverse_transition_block(x, 0.5, 'itb_2')
        x = tf_dense_block(x, 4, 'idb_2')

        x = tf_inverse_transition_block(x, 0.5, 'itb_3')
        x = tf_dense_block(x, 4, 'idb_3')

        x = tf_inverse_transition_block(x, 0.5, 'itb_4')
        x = tf_dense_block(x, 4, 'idb_4')

        self.reconstruction = Conv3D(dims[0], 3,
                                     activation=activation,
                                     data_format='channels_first',
                                     padding='SAME', name='reconstruction')(x)

        super(DenseAutoEncoder, self).__init__(
            optimiser, loss, opt_args
        )


class SingleLayerAutoEncoder(AutoEncoderBase):
    """Single layer nonconvolutional autoencoder."""

    def __init__(self,
                 dims,
                 encoding_size=10,
                 optimiser=tf.keras.optimizers.SGD,
                 loss='mse',
                 final_activation='sigmoid',
                 **opt_args):
        """Setup for autoencoder architecture."""

        activations = super(SingleLayerAutoEncoder, self)._define_activations()
        activation = activations.get(final_activation, 'linear')

        self.input_image = Input(shape=dims, dtype=tf.float32,
                                 name='input_image')
        x = Flatten()(self.input_image)
        self.encoding = Dense(
            encoding_size, name='encoding', activation='relu')(x)
        x = Dense(np.prod(dims),
                  activation=activation)(self.encoding)
        self.reconstruction = Reshape(dims, name='reconstruction')(x)

        super(SingleLayerAutoEncoder, self).__init__(
            optimiser, loss, opt_args
        )