"""
Created on Sat Jun 20 12:30:08 2020

@author: scantleb
@brief: AutoEncoder class definition

Autoencoders learn a mapping from a high dimensional space to a lower
dimensional space, as well as the inverse.
"""
from abc import abstractmethod
from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.models import Model

from layers import dense, residual
from layers.layer_functions import generate_activation_layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def define_discriminator(input_shape, hidden_size=None):
    """For placing adversarial constraints on latent distribution p(z|x).

    Arguments:
        input_shape: shape of the latent space
        hidden_size: size of the two hidden dense layers (a good starting point
            for this is the length of the latent vector)

    Returns:
        Uncompiled keras model which takes as input a latent vector (or
        vector sampled from a prior distribution) and outputs two numbers
        with the estimated probabilities of the input vector being sampled
        from the latent distribution (0) true prior distribution (1).
    """
    if hidden_size is None:
        hidden_size = input_shape[0]
    input_image = layers.Input(input_shape, name='input_distribution')
    x = layers.Dense(hidden_size)(input_image)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(hidden_size)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)

    probabilities = layers.Dense(
        2, activation='sigmoid', name='probabilities')(x)

    return Model(
        inputs=input_image, outputs=probabilities, name='discriminator')


class AutoEncoderBase(tf.keras.Model):
    """Abstract parent class for autoencoders."""

    def __init__(self,
                 dims,
                 encoding_size=10,
                 optimiser='sgd',
                 loss='mse',
                 hidden_activation='sigmoid',
                 encoding_activation='linear',
                 final_activation='sigmoid',
                 learning_rate_schedule=None,
                 metric_distance_threshold=-1.0,
                 adversarial=False,
                 adversarial_variance=10.0,
                 **opt_args):
        """Setup and compilation of autoencoder.

        Arguments:
            optimiser: any keras optimisation class
            loss: any keras loss fuction (or string reference), or or
                'unbalanced'/'composite_mse' (custom weighted loss functions)
            hidden_activation: activation function for hidden layers
            encoding_activation: activation function for encoding layer
            final_activation: activation function for reconstruction layer
            learning_rate_schedule: instance of class derived from
                LearningRateSchedule which can be called with the iteration
                number as an argument to give a learning rate
            opt_args: arguments for the keras optimiser (see keras
                documentation)
        """

        self.initialiser = tf.keras.initializers.HeNormal()  # weights init
        self.learning_rate_schedule = learning_rate_schedule
        self.encoding_size = encoding_size
        self.dims = dims
        self.iteration = 0
        self.adversarial = adversarial
        self.adversarial_variance = adversarial_variance
        self.encoding_activation = encoding_activation

        # Abstract method should be implemented in child class
        self.input_image, self.encoding, self.reconstruction = \
            self._construct_layers(
                dims=dims,
                encoding_size=encoding_size,
                hidden_activation=hidden_activation,
                final_activation=final_activation)

        # If optimiser is a string, turn it into a keras optimiser object
        if optimiser == 'adamw':
            optimiser = tfa.optimizers.AdamW
        elif optimiser == 'sgdw':
            optimiser = tfa.optimizers.SGDW
        elif isinstance(optimiser, str):
            optimiser = tf.keras.optimizers.get(optimiser).__class__

        inputs = [self.input_image]

        # Do we need to calculate and input the (expensive) distance-to-ligand
        # grid
        if loss == 'distance_mse' or metric_distance_threshold > 0:
            self.distances = layers.Input(
                shape=dims, dtype=tf.float32, name='distances')
            inputs.append(self.distances)

        # Composite mse requires an extra weight input
        if loss == 'composite_mse':
            self.frac = layers.Input(
                shape=(1,), dtype=tf.float32, name='frac')
            inputs.append(self.frac)

        super().__init__(
            inputs=inputs, outputs=[self.reconstruction, self.encoding],
            name='autoencoder')

        metrics = {
            'reconstruction': [mae, nonzero_mae, zero_mse,
                               nonzero_mse, trimmed_nonzero_mae,
                               trimmed_zero_mae]
        }

        if metric_distance_threshold > 0:
            self.add_metric(close_mae(self.input_image, self.reconstruction,
                                      self.distances,
                                      metric_distance_threshold),
                            name='close_mae')
            self.add_metric(
                close_nonzero_mae(
                    self.input_image, self.reconstruction, self.distances,
                    metric_distance_threshold),
                name='close_nonzero_mae')
            self.add_metric(
                close_zero_mae(self.input_image, self.reconstruction,
                               self.distances, metric_distance_threshold),
                name='close_zero_mae')

        if loss == 'composite_mse':
            self.add_loss(composite_mse(
                self.input_image, self.reconstruction, self.frac))
            self.compile(
                optimizer=optimiser(**opt_args),
                metrics=metrics
            )
        elif loss == 'distance_mse':
            self.add_loss(proximity_mse(
                self.input_image, self.reconstruction, self.distances))
            self.compile(
                optimizer=optimiser(**opt_args),
                metrics=metrics
            )
        elif loss == 'mse':
            self.compile(
                optimizer=optimiser(**opt_args),
                loss={'reconstruction': SquaredError(reduction='none'),
                      'encoding': None},
                metrics=metrics
            )
        else:
            self.compile(
                optimizer=optimiser(**opt_args),
                loss={'reconstruction': loss,
                      'encoding': None},
                metrics=metrics
            )

        if adversarial:
            encoder = tf.keras.models.Model(
                self.input_image, outputs=self.encoding, name='encoder')
            encoder.compile(
                optimizer=optimiser(**opt_args),
                loss={'encoding': 'binary_crossentropy'}
            )

            discriminator = define_discriminator(
                (encoding_size,), 2 * encoding_size)
            discriminator.compile(
                optimizer=optimiser(**opt_args),
                loss=disc_loss_fn,
                metrics=disc_loss_fn
            )

            self.encoder = encoder
            self.discriminator = discriminator
            self.train_step = tf.function(self.adversarial_train_step)
            self.train_function = None

    @abstractmethod
    def _construct_layers(self, dims, encoding_size, hidden_activation,
                          final_activation):
        """Setup for autoencoder architecture (abstract method).

        Arguments:
            dims: dimentionality of inputs
            encoding_size: size of bottleneck
            hidden_activation: activation function for hidden layers
            final_activation: activation function for final layer

        Returns:
            Tuple containing the input layer, the encoding layer, and the
            reconstruction layer of the autoencoder.

        Raises:
            NotImplementedError: if this method is not overridden by a class
                inheriting from this (abstract) class, or if this (abstract)
                class is initialised explicitly.
        """

        raise NotImplementedError('_construct_layers must be implemented '
                                  'in classes inherited from AutoEncoderBase. '
                                  'AutoEncoderBase is intended for use as an '
                                  'abstract class and should not be explicitly'
                                  ' instantiated.')

    def get_config(self):
        """Overloaded method; see base class (tf.keras.Model)."""
        config = super().get_config()
        config.update({'learning_rate_schedule': self.learning_rate_schedule,
                       'train_step': self.train_step})
        if self.adversarial:
            config.update({
                'adversarial_train_step': tf.function(
                    self.adversarial_train_step),
                'train_step': tf.function(self.adversarial_train_step)})
        try:
            d = self.get_layer('distances')
        except ValueError:
            pass
        else:
            config.update({'distances': d})
        try:
            config.update({'discriminator': self.discriminator})
            config.update({'encoder': self.encoder})
        except AttributeError:
            pass
        return config

    def adversarial_train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with backprop.GradientTape() as reconstruction_tape:
            reconstructions, latent_representations = self(x, training=True)
            frac = x.get('frac', sample_weight)
            reconstruction_loss = self.compiled_loss(
                y, reconstructions, frac,
                regularization_losses=self.losses)
        self.optimizer.minimize(
            reconstruction_loss, self.trainable_variables,
            tape=reconstruction_tape)
        self.compiled_metrics.update_state(y, reconstructions, sample_weight)
        generator_metrics = {m.name: m.result() for m in self.metrics}
        x = x.get('input_image')
        with backprop.GradientTape() as disc_tape:
            batch_size = x.shape[0]
            prior_sample = tf.random.normal(
                (batch_size, self.encoding_size), 0.0,
                np.sqrt(self.adversarial_variance), dtype=tf.float32,
                name='prior_sample'
            )
            latent_classifications = self.discriminator(
                latent_representations, training=True)
            prior_classifications = self.discriminator(
                prior_sample, training=True)
            disc_loss = self.discriminator.compiled_loss(
                prior_classifications, latent_classifications,
                sample_weight, regularization_losses=self.discriminator.losses)
        self.discriminator.optimizer.minimize(
            disc_loss, self.discriminator.trainable_variables, tape=disc_tape)
        self.discriminator.compiled_metrics.update_state(
            prior_classifications, latent_classifications, sample_weight)
        generator_metrics.update({'disc_{}'.format(m.name): m.result() for m in
                                  self.discriminator.metrics})

        with backprop.GradientTape() as enc_tape:
            encodings = self.encoder(x, training=True)
            classifications = self.discriminator(encodings, training=False)
            enc_loss = self.encoder.compiled_loss(
                tf.ones_like(classifications), classifications, sample_weight,
                regularization_losses=self.encoder.losses)
        self.encoder.optimizer.minimize(
            enc_loss, self.encoder.trainable_variables, tape=enc_tape)
        self.encoder.compiled_metrics.update_state(
            tf.ones_like(classifications), classifications)
        generator_metrics.update(
            {'gen_{}'.format(m.name): m.result() for m in self.encoder.metrics})

        mean, variance = tf.nn.moments(latent_representations, axes=[1])
        mean_mean = float(tf.reduce_mean(mean))
        mean_variance = float(tf.reduce_mean(variance))

        prior_means, prior_variances = tf.nn.moments(prior_sample, axes=[1])
        prior_mean = float(tf.reduce_mean(prior_means))
        prior_variance = float(tf.reduce_mean(prior_variances))

        # Let's get some measure of how good our discriminator/generator are
        real_probability = self.discriminator(prior_sample, training=False)
        fake_probability = self.discriminator(
            latent_representations, training=False)

        generator_metrics.update({
            'fake_prob': fake_probability,
            'real_prob': real_probability,
            'mean': mean_mean,
            'variance': mean_variance,
            'prior_mean': prior_mean,
            'prior_variance': prior_variance,
            'latent_representations': latent_representations
        })
        self.iteration += 1
        return generator_metrics


class DenseAutoEncoder(AutoEncoderBase):
    """Convolutional autoencoder with Dense connectivity."""

    def _construct_layers(self, dims, encoding_size, hidden_activation,
                          final_activation):
        """Overloaded method; see base class (AutoeEncoderBase)"""

        encoding_activation_layer = next(generate_activation_layers(
            'encoding', hidden_activation, append_name_info=False))
        decoding_activation_layer = next(generate_activation_layers(
            'decoding', hidden_activation, append_name_info=False))

        input_image = layers.Input(
            shape=dims, dtype=tf.float32, name='input_image')

        blocks = 8

        # Hidden layers
        x = dense.tf_dense_block(input_image, blocks, "db_1", hidden_activation)
        x = dense.tf_transition_block(x, 0.5, "tb_1", hidden_activation)

        x = dense.tf_dense_block(x, blocks, "db_2", hidden_activation)
        x = dense.tf_transition_block(x, 0.5, 'tb_2', hidden_activation)

        final_shape = x.shape
        x = layers.Flatten(data_format='channels_first')(x)

        x = layers.Dense(encoding_size, kernel_initializer=self.initialiser)(x)
        encoding = encoding_activation_layer(x)

        decoding = layers.Dense(reduce(mul, final_shape[1:]),
                                kernel_initializer=self.initialiser)(encoding)
        decoding = decoding_activation_layer(decoding)

        reshaped = layers.Reshape(final_shape[1:])(decoding)

        x = dense.tf_inverse_transition_block(
            reshaped, 0.5, 'itb_1', hidden_activation)
        x = dense.tf_dense_block(x, blocks, 'idb_1', hidden_activation)

        x = dense.tf_inverse_transition_block(
            x, 0.5, 'itb_2', hidden_activation)
        x = dense.tf_dense_block(x, blocks, 'idb_2', hidden_activation)

        reconstruction = layers.Conv3D(dims[0], 3,
                                       activation=final_activation,
                                       kernel_initializer=self.initialiser,
                                       data_format='channels_first',
                                       use_bias=False,
                                       padding='SAME', name='reconstruction')(x)

        return input_image, encoding, reconstruction


class SingleLayerAutoEncoder(AutoEncoderBase):
    """Single layer nonconvolutional autoencoder."""

    def _construct_layers(self, dims, encoding_size, hidden_activation,
                          final_activation):
        """Overloaded method; see base class (AutoeEncoderBase)"""

        encoding_activation_layer = next(generate_activation_layers(
            'encoding', hidden_activation, append_name_info=False))

        input_image = layers.Input(shape=dims, dtype=tf.float32,
                                   name='input_image')
        x = layers.Flatten()(input_image)

        x = layers.Dense(encoding_size)(x)
        encoding = encoding_activation_layer(x)

        x = layers.Dense(np.prod(dims),
                         activation=final_activation)(encoding)
        reconstruction = layers.Reshape(dims, name='reconstruction')(x)

        return input_image, encoding, reconstruction


class MultiLayerAutoEncoder(AutoEncoderBase):
    """Three layered convolutional autoencoder."""

    def _construct_layers(self, dims, encoding_size, hidden_activation,
                          final_activation):
        """Overloaded method; see base class (AutoeEncoderBase)"""

        conv_activation = generate_activation_layers(
            'enc_conv', hidden_activation, append_name_info=True)

        input_image = layers.Input(shape=dims, dtype=tf.float32,
                                   name='input_image')

        conv_args = {'padding': 'same',
                     'data_format': 'channels_first',
                     'use_bias': False, 'kernel_initializer': self.initialiser}

        bn = lambda x: layers.BatchNormalization(axis=1, epsilon=1.001e-5)(x)

        x = layers.Conv3D(1024, 3, 2, **conv_args)(input_image)
        x = next(conv_activation)(x)
        x = bn(x)

        x = layers.Conv3D(1024, 3, 2, **conv_args)(x)
        x = next(conv_activation)(x)
        x = bn(x)

        x = layers.Conv3D(1024, 3, 2, **conv_args)(x)
        x = next(conv_activation)(x)
        x = bn(x)

        final_shape = x.shape[1:]

        x = layers.Flatten(data_format='channels_first', name='enc_flatten')(x)

        x = layers.Dense(encoding_size)(x)

        encoding = next(generate_activation_layers(
            'encoding', self.encoding_activation, append_name_info=False))(x)

        x = layers.Dense(np.prod(final_shape), name='dec_dense')(encoding)
        x = next(conv_activation)(x)
        x = layers.Reshape(final_shape, name='dec_reshape')(x)

        x = layers.Conv3DTranspose(1024, 3, 2, **conv_args)(x)
        x = next(conv_activation)(x)
        x = bn(x)

        x = layers.Conv3DTranspose(1024, 3, 2, **conv_args)(x)
        x = next(conv_activation)(x)
        x = bn(x)

        reconstruction = layers.Conv3DTranspose(
            dims[0], 3, 2, activation=final_activation, name='reconstruction',
            **conv_args)(x)

        return input_image, encoding, reconstruction


class ResidualAutoEncoder(AutoEncoderBase):
    """Convolutional autoencoder with residual connections."""

    def _construct_layers(self, dims, encoding_size, hidden_activation,
                          final_activation):
        """Overloaded method; see base class (AutoeEncoderBase)"""

        conv_args = {'padding': 'same',
                     'data_format': 'channels_first',
                     'use_bias': False}

        conv_activation = generate_activation_layers(
            'conv', hidden_activation, append_name_info=True)

        encoding_activation_layer = next(generate_activation_layers(
            'encoding', hidden_activation, append_name_info=False))

        input_image = layers.Input(shape=dims, dtype=tf.float32,
                                   name='input_image')

        conv_blocks = 2
        bn_axis = 1

        x = layers.Conv3D(64, 3, 1, name='init_conv', **conv_args)(input_image)

        x = residual.ResBlock(
            conv_blocks, 32, 3, 2, hidden_activation, 'res_1_1')(x)
        x = residual.ResBlock(
            conv_blocks, 32, 3, 1, hidden_activation, 'res_1_2')(x)
        x = residual.ResBlock(
            conv_blocks, 32, 3, 1, hidden_activation, 'res_1_3')(x)

        x = residual.ResBlock(
            conv_blocks, 64, 3, 2, hidden_activation, 'res_2_1')(x)
        x = residual.ResBlock(
            conv_blocks, 64, 3, 1, hidden_activation, 'res_2_2')(x)
        x = residual.ResBlock(
            conv_blocks, 64, 3, 1, hidden_activation, 'res_2_3')(x)

        x = residual.ResBlock(
            conv_blocks, 64, 3, 2, hidden_activation, 'res_3_1')(x)

        final_shape = x.shape[1:]

        x = layers.Flatten(data_format='channels_first')(x)

        x = layers.Dense(encoding_size)(x)
        encoding = encoding_activation_layer(x)

        x = layers.Dense(np.prod(final_shape))(encoding)
        x = next(conv_activation)(x)
        x = layers.Reshape(final_shape)(x)

        x = layers.Conv3DTranspose(128, 3, 2, **conv_args)(x)

        x = next(conv_activation)(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5)(x)

        x = layers.Conv3DTranspose(512, 1, 1, **conv_args)(x)
        x = next(conv_activation)(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5)(x)

        x = layers.Conv3DTranspose(512, 1, 1, **conv_args)(x)
        x = next(conv_activation)(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5)(x)

        x = layers.Conv3DTranspose(256, 3, 2, **conv_args)(x)
        x = next(conv_activation)(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5)(x)

        x = layers.Conv3DTranspose(256, 3, 1, **conv_args)(x)
        x = next(conv_activation)(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5)(x)

        x = layers.Conv3DTranspose(256, 3, 1, **conv_args)(x)
        x = next(conv_activation)(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5)(x)

        reconstruction = layers.Conv3DTranspose(dims[0], 3, 2,
                                                name='reconstruction',
                                                activation=final_activation,
                                                **conv_args)(x)

        return input_image, encoding, reconstruction


class SquaredError(tf.keras.losses.Loss):
    """Implementation of the squared error with no reduction.

    The keras implementation of the squared error appears to reduce along the
    last axis, even with reduction='none' is specified. This is not the case for
    this SquaredError class.
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO,
                 name='squared_error', reduction_axis=None):
        super().__init__(reduction=reduction, name=name)
        reduction_axis = reduction_axis if reduction_axis is not None else []
        self.reduction_axis = reduction_axis

    def call(self, y_true, y_pred):
        """Overridden method; see base class (tf.keras.loss.Loss).

        Get the point-wise squared difference between two tensors.

        Arguments:
            y_true: input tensor
            y_pred: output tensor of the autoencoder (reconstruction)
        """
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.cast(y_pred, y_true.dtype)
        return tf.reduce_mean(tf.square(y_true - y_pred),
                              axis=self.reduction_axis)


def nonzero_mse(target, reconstruction):
    """Mean squared error for non-zero values in the target matrix

    Finds the mean squared error for all parts of the input tensor which
    are not equal to zero.

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder

    Returns:
        Mean squared error for all non-zero entries in the target
    """
    mask = tf.cast(tf.not_equal(target, 0), tf.float32)
    masked_difference = (target - reconstruction) * mask
    return tf.square(masked_difference)


def zero_mse(target, reconstruction):
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
    return tf.square(masked_difference)


def composite_mse(target, reconstruction, ratio):
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
    # Reduce_mean required for broadcast shape reasons
    frac = tf.reduce_mean(tf.divide(ratio, 1. + ratio))
    return tf.math.add(
        tf.math.multiply(frac, trimmed_nonzero_mse(target, reconstruction)),
        tf.math.multiply(1. - frac, trimmed_zero_mse(target, reconstruction)))


def proximity_mse(target, reconstruction, distances):
    """Weighted mean squared error by proximity to ligand density.

    Finds the MSE between the target and reconstruction, weighted by the inverse
    of the distance between each point on the grid and the nearest point which
    contains a non-zero input in an ligand channel. This distances grid can be
    found using calcul_distances.calculate_distances.

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder
        distances: grid of the same dimension as target containing distances
            between each point and the nearest point with ligand density. This
            should be constructed by stacking n_channels copies of the 3D output
            of calculate_distances.calculate_distances on a new axis at position
            1.

    Returns:
        4D tensor of the same shape as the input tensors. The mean squared error
        at each point is weighted by the distances according to the following
        expression in order to avoid dividing by zero errors:

            weighted_squared_error = squared_error * (1 / (distance**2 + 0.5)
    """
    proximities = 0.5 + 0.5 * (tf.math.tanh(4.0 - distances))
    squared_difference = tf.math.squared_difference(target, reconstruction)
    masked_sq_difference = tf.math.multiply(proximities, squared_difference)
    return masked_sq_difference


def close_mae(target, reconstruction, distances, threshold):
    """Mean average error thresholded by proximity to ligand density.

    Finds the MAE for inputs which are closer than a threshold distance from
    any ligand channel with a value greater than zero. This should only be used
    as a metric (not a loss function).

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder
        distances: grid of the same dimension as target containing distances
            between each point and the nearest point with ligand density. This
            should be constructed by stacking n_channels copies of the 3D output
            of gnina_tensorflow_cpp.calculate_distances on a new axis at
            position 1.
        threshold: threshold distance (Angstroms)

    Returns:
        Mean average error for values within <threshold> angstroms of a nonzero
        ligand channel input.
    """
    mask = tf.cast(tf.math.less_equal(distances, float(threshold)), tf.float32)
    mask_sum = tf.reduce_sum(mask)
    difference = tf.math.abs(target - reconstruction)
    masked_difference = tf.multiply(mask, difference)
    return tf.reduce_sum(masked_difference) / mask_sum


def close_nonzero_mae(target, reconstruction, distances, threshold):
    """MAE thresholded by proximity to ligand density, for non-zero inputs.

    Finds the MAE for non zero inputs which are closer than a threshold distance
    from any ligand channel with a value greater than zero. This should only be
    used as a metric (not a loss function).

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder
        distances: grid of the same dimension as target containing distances
            between each point and the nearest point with ligand density. This
            should be constructed by stacking n_channels copies of the 3D output
            of gnina_tensorflow_cpp.calculate_distances on a new axis at
            position 1.
        threshold: threshold distance (Angstroms)

    Returns:
        Mean average error for values within <threshold> angstroms of a nonzero
        ligand channel input, and which have values greater than zero.
    """
    dist_mask = tf.cast(
        tf.math.less_equal(distances, float(threshold)), tf.float32)
    nonzero_mask = tf.cast(tf.math.not_equal(target, 0.0), tf.float32)
    mask = tf.multiply(dist_mask, nonzero_mask)
    mask_sum = tf.reduce_sum(mask)
    difference = tf.math.abs(target - reconstruction)
    masked_difference = tf.multiply(mask, difference)
    return tf.reduce_sum(masked_difference) / mask_sum


def close_zero_mae(target, reconstruction, distances, threshold):
    """MAE thresholded by proximity to ligand density, for inputs equal to zero.

    Finds the MAE for inputs equal to zero which are closer than a threshold
    distance from any ligand channel with a value greater than zero. This should
    only be used as a metric (not a loss function).

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder
        distances: grid of the same dimension as target containing distances
            between each point and the nearest point with ligand density. This
            should be constructed by stacking n_channels copies of the 3D output
            of gnina_tensorflow_cpp.calculate_distances on a new axis at
            position 1.
        threshold: threshold distance (Angstroms)

    Returns:
        Mean average error for values within <threshold> angstroms of a nonzero
        ligand channel input, and which have values equal to zero.
    """
    dist_mask = tf.cast(
        tf.math.less_equal(distances, float(threshold)), tf.float32)
    nonzero_mask = tf.cast(tf.math.equal(target, 0.0), tf.float32)
    mask = tf.multiply(dist_mask, nonzero_mask)
    mask_sum = tf.reduce_sum(mask)
    difference = tf.math.abs(target - reconstruction)
    masked_difference = tf.multiply(mask, difference)
    return tf.reduce_sum(masked_difference) / mask_sum


def mae(target, reconstruction):
    """Mean absolute error loss function.

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder

    Returns:
        Tensor containing the mean absolute error between the target and
        the reconstruction.
    """
    return tf.reduce_mean(tf.abs(target - reconstruction))


def zero_mae(target, reconstruction):
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
    mask_sum = tf.reduce_sum(mask)
    abs_diff = tf.abs(target - reconstruction)
    masked_abs_diff = tf.math.multiply(abs_diff, mask)
    return tf.reduce_sum(masked_abs_diff) / mask_sum


def nonzero_mae(target, reconstruction):
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
    mask = tf.cast(tf.not_equal(target, 0), tf.float32)
    abs_diff = tf.abs(target - reconstruction)
    masked_abs_diff = tf.math.multiply(abs_diff, mask)
    mask_sum = tf.reduce_sum(mask)
    return tf.reduce_sum(masked_abs_diff) / mask_sum


def trimmed_nonzero_mae(target, reconstruction):
    """Mean absolute error loss function target values are not zero.

    Inputs less than two units from the edge of the bounding cube are discarded.

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder

    Returns:
        Tensor containing the mean absolute error between the target and
        the reconstruction where the mean is taken over values where
        the target is not zero.
        This can be NaN if there are no nonzero inputs.
    """
    _, _, x, y, z = target.shape
    begin = [0, 0, 3, 3, 3]
    end = [-1, -1, x - 3, y - 3, z - 3]
    trimmed_target = tf.slice(target, begin, end)
    trimmed_reconstruction = tf.slice(reconstruction, begin, end)
    return nonzero_mae(trimmed_target, trimmed_reconstruction)


def trimmed_zero_mae(target, reconstruction):
    """Mean absolute error loss function target values are zero.

    Inputs less than two units from the edge of the bounding cube are discarded.

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder

    Returns:
        Tensor containing the mean absolute error between the target and
        the reconstruction where the mean is taken over values where
        the target is equal to zero.
        This can be NaN if there are no inputs equal to zero.
    """
    _, _, x, y, z = target.shape
    begin = [0, 0, 3, 3, 3]
    end = [-1, -1, x - 3, y - 3, z - 3]
    trimmed_target = tf.slice(target, begin, end)
    trimmed_reconstruction = tf.slice(reconstruction, begin, end)
    return zero_mae(trimmed_target, trimmed_reconstruction)


def trimmed_nonzero_mse(target, reconstruction):
    """Mean squared error for non-zero values in the target matrix

    Finds the mean squared error for all parts of the input tensor which
    are not equal to zero. Inputs less than two units from the edge of the
    bounding cube are discarded.

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder

    Returns:
        Mean squared error for all non-zero entries in the target
    """
    _, _, x, y, z = target.shape
    begin = [0, 0, 2, 2, 2]
    end = [-1, -1, x - 2, y - 2, z - 2]
    trimmed_target = tf.slice(target, begin, end)
    trimmed_reconstruction = tf.slice(reconstruction, begin, end)
    return nonzero_mse(trimmed_target, trimmed_reconstruction)


def trimmed_zero_mse(target, reconstruction):
    """Mean squared error for non-zero values in the target matrix

    Finds the mean squared error for all parts of the input tensor which
    are not equal to zero. Inputs less than two units from the edge of the
    bounding cube are discarded.

    Arguments:
        target: input tensor
        reconstruction: output tensor of the autoencoder

    Returns:
        Mean squared error for all non-zero entries in the target
    """
    _, _, x, y, z = target.shape
    begin = [0, 0, 2, 2, 2]
    end = [-1, -1, x - 2, y - 2, z - 2]
    trimmed_target = tf.slice(target, begin, end)
    trimmed_reconstruction = tf.slice(reconstruction, begin, end)
    return zero_mse(trimmed_target, trimmed_reconstruction)


def disc_loss_fn(prior_classifications, latent_classifications):
    prior_loss = crossentropy(
        tf.ones_like(prior_classifications), prior_classifications)
    latent_loss = crossentropy(
        tf.zeros_like(latent_classifications), latent_classifications)
    return (prior_loss + latent_loss) / 2


def enc_loss_fn(latent_classifications):
    enc_loss = crossentropy(
        tf.ones(latent_classifications), latent_classifications)
    return enc_loss
