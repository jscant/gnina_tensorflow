"""
Created on Sat Jun 20 12:30:08 2020

@author: scantleb
@brief: AutoEncoder class definition

Autoencoders learn a mapping from a high dimensional space to a lower
dimensional space, as well as the inverse.
"""
from abc import abstractmethod, ABC
from pathlib import Path

import molgrid
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from gnina_tensorflow_cpp import calculate_distances as cd
from tensorflow.keras import layers

from layers import dense_callable
from layers.dense_callable import DenseBlock, TransitionBlock, \
    InverseTransitionBlock
from layers.layer_functions import generate_activation_layers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
epsilon = tf.constant(1e-6, dtype=tf.float32)


class Discriminator(tf.keras.Model):

    def __init__(self, input_shape):
        super().__init__(name='discriminator')
        self.disc = False
        self.inp_shape = input_shape
        self.input_image = layers.Input(
            shape=input_shape, dtype=tf.float32, name='input_image')

        self.mp_1 = layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same',
            data_format='channels_first')

        self.conv_1 = layers.Conv3D(
            32, 3, activation='relu', padding='same', use_bias=False,
            data_format='channels_first')

        self.dense_layers = []
        self.dense_args = []

        dummy = np.zeros((1,) + input_shape, dtype='float32')
        x = self.mp_1(dummy)
        x = self.conv_1(x)
        for i in range(3):
            final = True if i == 2 else False
            self.dense_layers.append(
                dense_callable.DenseBlock(4, 'db_{}'.format(i + 1)))
            x = self.dense_layers[-1](x)
            self.dense_layers.append(dense_callable.TransitionBlock(
                x.shape, 1.0, 'tb_{}'.format(i + 1), final=final))

        self.global_max = layers.GlobalMaxPooling3D(
            data_format='channels_first')

        self.probability = layers.Dense(
            1, activation='sigmoid', name='probabilities')
        self.opt = tf.keras.optimizers.SGD(
            lr=0.01, momentum=True, nesterov=True)

    def call(self, inputs, training=None, mask=None):
        x = self.mp_1(inputs)
        x = self.conv_1(x)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.global_max(x)
        return self.probability(x)

    def summary(self, line_length=None, positions=None, print_fn=None):
        return self._model().summary(line_length, positions, print_fn)

    def _model(self):
        x = layers.Input(self.inp_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        pass


def fake_quality(discriminator_predictions):
    similarity = crossentropy(tf.ones_like(discriminator_predictions),
                              discriminator_predictions)
    return similarity


def discriminator_loss_fn(real_predictions, fake_predictions):
    real_loss = crossentropy(tf.ones_like(real_predictions), real_predictions)
    fake_loss = crossentropy(tf.zeros_like(fake_predictions), fake_predictions)
    return 0.5 * (real_loss + fake_loss)


class AutoEncoderBase(tf.keras.Model):
    """Abstract parent class for autoencoders."""

    def __init__(self, dims, batch_size, train_types, data_root,
                 latent_size=10, optimiser='sgd', loss='mse',
                 hidden_activation='sigmoid', final_activation='sigmoid',
                 learning_rate_schedule=None, metric_distance_threshold=-1.0,
                 ligmap=None, recmap=None, dimension=23.5, resolution=0.5,
                 binary_mask=False, adversarial=False, **opt_args):
        """Setup and compilation of autoencoder.

        Arguments:
            optimiser: any keras optimisation class
            loss: any keras loss fuction (or string reference), or or
                'unbalanced'/'composite_mse' (custom weighted loss functions)
            hidden_activation: activation function for hidden layers
            final_activation: activation function for reconstruction layer
            learning_rate_schedule: instance of class derived from
                LearningRateSchedule which can be called with the iteration
                number as an argument to give a learning rate
            adversarial: use convolutional neural network discriminator to
                force reconstructions to be more `realistic`
            opt_args: arguments for the keras optimiser (see keras
                documentation)
        """
        super().__init__(name='Autoencoder')
        self.metric_distance_threshold = metric_distance_threshold
        self.resolution = resolution
        self.dimension = dimension
        self.dims = dims
        self.latent_size = latent_size
        self.initialiser = tf.keras.initializers.HeNormal()  # weights init
        self.learning_rate_schedule = learning_rate_schedule
        self.encoding_layers = []
        self.decoding_layers = []
        self.batch_size = batch_size

        if adversarial:
            self.discriminator = Discriminator(dims)
            self.get_gradients = self._get_gradients_adversarial
        else:
            self.discriminator = None
            self.get_gradients = self._get_gradients

        # Abstract method should be implemented in child class
        self._construct_layers(
            dims=dims,
            latent_size=latent_size,
            hidden_activation=hidden_activation,
            final_activation=final_activation)

        self.metrics_dict = {
            'reconstruction': {
                name: func for name, func in zip(
                    ['mae', 'nonzero_mae', 'zero_mae'],
                    [mae, nonzero_mae, zero_mae])
            }
        }

        loss_synonyms = {'mean_squared_error': 'MeanSquaredError',
                         'mae': 'MeanAbsoluteError',
                         'binary_crossentropy': 'BinaryCrossentropy',
                         'categorical_crossentropy': 'CategoricalCrossentropy',
                         'hinge': 'Hinge',
                         'squared_hinge': 'SquaredHinge'}

        loss_reduction = tf.keras.losses.Reduction.AUTO
        reduction_axis = [-4, -3, -2, -1]
        self.loss = None
        if loss == 'mse':
            loss_class = SquaredError
        elif loss == 'distance_mse':
            loss_class = ProximityMse
        elif loss == 'composite_mse':
            loss_class = CompositeMse
        elif loss == 'binary_crossentropy':
            loss_class = BinaryCrossentropy
        else:
            loss_class = tf.keras.losses.get(
                loss_synonyms.get(loss, loss)).__class__
            self.loss = loss_class(reduction=loss_reduction)
        if self.loss is None:
            self.loss = loss_class(
                reduction=loss_reduction, reduction_axis=reduction_axis)

        example_provider_kwargs = {
            'data_root': str(Path(data_root).expanduser()), 'balanced': False,
            'shuffle': True, 'cache_structs': False
        }
        if ligmap is None or recmap is None:
            # noinspection PyArgumentList
            self.e = molgrid.ExampleProvider(
                **example_provider_kwargs
            )
            self.rec_channels = 14
        else:
            rec_typer = molgrid.FileMappedGninaTyper(recmap)
            lig_typer = molgrid.FileMappedGninaTyper(ligmap)
            self.e = molgrid.ExampleProvider(
                rec_typer, lig_typer, **example_provider_kwargs)
            with open(recmap, 'r') as f:
                self.rec_channels = len(
                    [line for line in f.readlines() if len(line)])
        self.e.populate(str(Path(train_types).expanduser()))

        # noinspection PyArgumentList
        self.gmaker = molgrid.GridMaker(
            binary=binary_mask,
            dimension=dimension,
            resolution=resolution)

        # noinspection PyArgumentList
        input_shape = self.gmaker.grid_dimensions(self.e.num_types())
        tensor_shape = (self.batch_size,) + input_shape
        self.input_tensor = molgrid.MGrid5f(*tensor_shape)

        self.iteration = 0

        # If optimiser is a string, turn it into a keras optimiser object
        if optimiser == 'adamw':
            optimiser = tfa.optimizers.AdamW
        elif optimiser == 'sgdw':
            optimiser = tfa.optimizers.SGDW
        elif isinstance(optimiser, str):
            optimiser = tf.keras.optimizers.get(optimiser).__class__
        self.opt = optimiser(**opt_args)

    @abstractmethod
    def _construct_layers(self, dims, latent_size, hidden_activation,
                          final_activation):
        """Setup for autoencoder architecture (abstract method).

        Arguments:
            dims: dimentionality of inputs
            latent_size: size of bottleneck
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

    @tf.function
    def _get_gradients(self, input_images):
        with tf.GradientTape() as tape:
            feature_vectors = [input_images]
            if isinstance(self.loss, ProximityMse):
                spatial_information = cd(
                    self.rec_channels, np.asfortranarray(input_images),
                    self.resolution)
                feature_vectors.append(spatial_information)

            reconstructions = self.call(input_images, training=True)

            if len(feature_vectors) == 1:
                reconstruction_loss = self.loss(
                    feature_vectors[0], reconstructions)
            else:
                reconstruction_loss = self.loss(
                    feature_vectors[0], reconstructions, feature_vectors[1])

            metrics = {}
            for name, func in self.metrics_dict.get('reconstruction').items():
                metrics[name] = float(func(input_images, reconstructions))

        grads = tape.gradient(
            reconstruction_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(
            grads, self.trainable_variables
        ))
        return reconstruction_loss, metrics

    @tf.function
    def _get_gradients_adversarial(self, input_images):
        """
        The tf.function decoration causes this function to be executed in graph
        mode. This has the effect of a massive increase in the speed of training
        except when using the distance_mse loss option which is significantly
        slowed down.
        """
        with tf.GradientTape() as discriminator_tape, \
                tf.GradientTape() as sim_tape, \
                tf.GradientTape() as recon_tape:
            feature_vectors = [input_images]
            if isinstance(self.loss, ProximityMse):
                spatial_information = cd(
                    self.rec_channels, np.asfortranarray(input_images),
                    self.resolution)
                feature_vectors.append(spatial_information)

            reconstructions = self(input_images, training=True)

            probabilities_real = self.discriminator(
                input_images, training=True)
            probabilities_fake = self.discriminator(
                reconstructions, training=True)

            if len(feature_vectors) == 1:
                reconstruction_loss = self.loss(
                    feature_vectors[0], reconstructions)
            else:
                reconstruction_loss = self.loss(
                    feature_vectors[0], reconstructions, feature_vectors[1])

            similarity_loss = tf.divide(fake_quality(
                probabilities_fake), 1000)
            discriminator_loss = discriminator_loss_fn(
                probabilities_real, probabilities_fake)

            metrics = {}
            for name, func in self.metrics_dict.get('reconstruction').items():
                metrics[name] = float(func(input_images, reconstructions))
            metrics['discriminator_loss'] = discriminator_loss
            metrics['similarity_loss'] = similarity_loss
            metrics['real_probabilities'] = tf.reduce_mean(probabilities_real)
            metrics['fake_probabilities'] = tf.reduce_mean(probabilities_fake)

        gen_recon_grads = recon_tape.gradient(
            reconstruction_loss, self.trainable_variables)

        if reconstruction_loss < 0.005:
            gen_fake_grads = sim_tape.gradient(
                similarity_loss, self.trainable_variables
            )
            discriminator_grads = discriminator_tape.gradient(
                discriminator_loss, self.discriminator.trainable_variables)
            self.opt.apply_gradients(zip(
                gen_fake_grads, self.trainable_variables
            ))
            self.discriminator.opt.apply_gradients(zip(
                discriminator_grads, self.discriminator.trainable_variables
            ))

        self.opt.apply_gradients(zip(
            gen_recon_grads, self.trainable_variables
        ))

        return reconstruction_loss, metrics

    def training_step(self):
        """Perform backpropagation on a single batch.

        Returns:
            Tuple: the reconstruction loss and a dictionary containing a mapping
            from metric names to the value of that metric for the batch.
        """
        batch = self.e.next_batch(self.batch_size)
        self.gmaker.forward(batch, self.input_tensor, 0,
                            random_rotation=True)
        input_numpy = self.input_tensor.tonumpy()
        original_images = tf.convert_to_tensor(np.minimum(1.0, input_numpy))
        self.iteration += 1
        return self.get_gradients(original_images)

    def get_config(self):
        """Overloaded method; see base class (tf.keras.Model)."""
        config = super().get_config()
        config.update(
            {'metric_distance_threshold': self.metric_distance_threshold,
             'resolution': self.resolution,
             'dimension': self.dimension,
             'dims': self.dims,
             'latent_size': self.latent_size,
             'initialiser': self.initialiser,
             'learning_rate_schedule': self.learning_rate_schedule,
             'encoding_layers': self.encoding_layers,
             'decoding_layers': self.decoding_layers,
             'batch_size': self.batch_size,
             'mae': mae,
             'trimmed_nonzero_mae': trimmed_nonzero_mae,
             'zero_mse': zero_mse,
             'nonzero_mse': nonzero_mse,
             '_construct_layers': self._construct_layers,
             'loss': self.loss,
             'e': self.e,
             'rec_channels': self.rec_channels,
             'gmaker': self.gmaker,
             'input_tensor': self.input_tensor,
             'iteration': self.iteration,
             'opt': self.opt,
             'train_step': self.train_step,
             '_get_gradients': self._get_gradients}
        )
        return config

    def _find_final_shape(self):
        """Finds output shape of final convolutional layer in encoder."""
        dummy_input = np.zeros((self.batch_size,) + self.dims, dtype='float32')
        for layer in self.encoding_layers:
            dummy_input = layer(dummy_input)
        return dummy_input.shape[1:]

    def call(self, inputs, training=None, mask=None):
        """Overridden method; see base class (tf.keras.Model)."""
        latent_representation = self.encode(inputs, training=training)
        reconstruction = self.decode(latent_representation, training=training)
        return reconstruction

    def encode(self, inputs, training=None):
        """Take input and return lower dimensional representation.

        Arguments:
            inputs: 5D tensor or numpy array
            training: are we running in training mode or for inference?
        """
        x = inputs
        for layer in self.encoding_layers:
            x = layer(x, training=training)
        return x

    def decode(self, inputs, training=None):
        """Take input and return lower dimensional representation.

        Arguments:
            inputs: 5D tensor or numpy array
            training: are we running in training mode or for inference?
        """
        x = inputs
        for layer in self.decoding_layers:
            x = layer(x, training=training)
        return x

    def summary(self, line_length=None, positions=None, print_fn=None):
        """See tf.keras.Model.summary() documentation for args.

        Prints out a summary of layers in the model."""
        return self._model().summary(line_length, positions, print_fn)

    def _model(self):
        """Returns a model where input and output shapes are determined."""
        x = layers.Input(self.dims)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def plot(self, to_file='model.png', show_shapes=False,
             show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96):
        """See tf.keras.plot_model() documentation for args.

        Saves a graphical representation of model to disk."""
        return tf.keras.utils.plot_model(
            self._model(), to_file=to_file, show_shapes=show_shapes,
            show_layer_names=show_layer_names, rankdir=rankdir,
            expand_nested=expand_nested, dpi=dpi)


class MultiLayerAutoEncoder(AutoEncoderBase):
    """Single layer nonconvolutional autoencoder."""

    def _construct_layers(self, dims, latent_size, hidden_activation,
                          final_activation):
        """Overloaded method; see base class (AutoeEncoderBase)"""
        encoding_activation_layer = next(generate_activation_layers(
            'encoding', hidden_activation, append_name_info=False))

        conv_activation = generate_activation_layers(
            'conv', hidden_activation, append_name_info=True)

        bn = lambda: layers.BatchNormalization(axis=1, epsilon=1.001e-5)

        conv_args = {'padding': 'same',
                     'data_format': 'channels_first',
                     'use_bias': False,
                     'kernel_initializer': self.initialiser}

        self.encoding_layers.append(layers.Conv3D(128, 3, 2, **conv_args))
        self.encoding_layers.append(next(conv_activation))
        self.encoding_layers.append(bn())

        self.encoding_layers.append(layers.Conv3D(256, 3, 2, **conv_args))
        self.encoding_layers.append(next(conv_activation))
        self.encoding_layers.append(bn())

        self.encoding_layers.append(layers.Conv3D(512, 3, 2, **conv_args))
        self.encoding_layers.append(next(conv_activation))
        self.encoding_layers.append(bn())

        # self.encoding_layers.append(layers.Conv3D(2048, 1, 1, **conv_args))
        # self.encoding_layers.append(next(conv_activation))
        # self.encoding_layers.append(bn())

        final_shape = self._find_final_shape()

        self.encoding_layers.append(
            layers.Flatten(data_format='channels_first'))

        self.encoding_layers.append(layers.Dense(latent_size))
        self.encoding_layers.append(encoding_activation_layer)

        self.decoding_layers.append(layers.Dense(np.prod(final_shape)))
        self.decoding_layers.append(next(conv_activation))
        self.decoding_layers.append(layers.Reshape(final_shape))

        # self.decoding_layers.append(
        #    layers.Conv3DTranspose(512, 1, 1, **conv_args))
        # self.decoding_layers.append(next(conv_activation))
        # self.decoding_layers.append(bn())

        self.decoding_layers.append(
            layers.Conv3DTranspose(256, 3, 2, **conv_args))
        self.decoding_layers.append(next(conv_activation))
        self.decoding_layers.append(bn())

        self.decoding_layers.append(
            layers.Conv3DTranspose(128, 3, 2, **conv_args))
        self.decoding_layers.append(next(conv_activation))
        self.decoding_layers.append(bn())

        self.decoding_layers.append(layers.Conv3DTranspose(
            dims[0], 3, 2, name='reconstruction', activation=final_activation,
            **conv_args))


class SingleLayerAutoEncoder(AutoEncoderBase):
    """Single layer nonconvolutional autoencoder."""

    def _construct_layers(self, dims, latent_size, hidden_activation,
                          final_activation):
        """Overloaded method; see base class (AutoeEncoderBase)"""

        encoding_activation_layer = next(generate_activation_layers(
            'encoding', hidden_activation, append_name_info=False))

        self.encoding_layers.append(layers.Flatten())

        self.encoding_layers.append(layers.Dense(latent_size))
        self.encoding_layers.append(encoding_activation_layer)

        self.decoding_layers.append(layers.Dense(
            np.prod(dims), activation=final_activation))
        self.decoding_layers.append(
            layers.Reshape(dims, name='reconstruction'))


class DenseAutoEncoder(AutoEncoderBase):
    """Convolutional autoencoder with layers.Dense connectivity."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _construct_layers(self, dims, latent_size, hidden_activation,
                          final_activation):
        """Overridden method; see base class (AutoeEncoderBase)"""

        encoding_activation_layer = next(generate_activation_layers(
            'encoding', hidden_activation, append_name_info=False))
        decoding_activation_layer = next(generate_activation_layers(
            'decoding', hidden_activation, append_name_info=False))

        blocks = 8

        # Hidden layers
        x = np.zeros((self.batch_size,) + self.dims, dtype='float32')
        for i in range(3):
            final = True if i == 2 else False
            self.encoding_layers.append(DenseBlock(
                blocks, 'encode_db_{}'.format(i + 1),
                activation=hidden_activation))
            x = self.encoding_layers[-1](x)
            self.encoding_layers.append(TransitionBlock(
                x.shape, 0.5, 'encode_tb_{}'.format(i + 1), final=final,
                activation=hidden_activation))
            x = self.encoding_layers[-1](x)

        final_shape = self._find_final_shape()

        self.encoding_layers.append(
            layers.Flatten(data_format='channels_first'))

        self.encoding_layers.append(
            layers.Dense(latent_size, kernel_initializer=self.initialiser))
        self.encoding_layers.append(encoding_activation_layer)

        self.decoding_layers.append(layers.Dense(
            np.prod(final_shape), kernel_initializer=self.initialiser))
        self.decoding_layers.append(decoding_activation_layer)

        self.decoding_layers.append(layers.Reshape(final_shape))

        x = np.zeros((self.batch_size,) + final_shape, dtype='float32')
        for i in range(2):
            self.decoding_layers.append(
                DenseBlock(blocks, 'decode_db_{}'.format(i + 1),
                           activation=hidden_activation))
            x = self.decoding_layers[-1](x)
            self.decoding_layers.append(InverseTransitionBlock(
                x.shape, 1.0, 'decode_itb_{}'.format(i + 1),
                activation=hidden_activation))
            x = self.decoding_layers[-1](x)

        self.decoding_layers.append(
            layers.Conv3D(dims[0], 3, activation=final_activation,
                          kernel_initializer=self.initialiser,
                          data_format='channels_first', use_bias=False,
                          padding='SAME', name='reconstruction'))


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


class ProximityMse(tf.keras.losses.Loss, ABC):
    """Weighted mean squared error by proximity to ligand density.

    Finds the MSE between the target and reconstruction, weighted by the inverse
    of the distance between each point on the grid and the nearest point which
    contains a non-zero input in an ligand channel. This distances grid can be
    found using calcul_distances.calculate_distances.
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO,
                 name='proximity_mse', reduction_axis=None):
        super().__init__(reduction=reduction, name=name)
        reduction_axis = reduction_axis if reduction_axis is not None else []
        self.reduction_axis = reduction_axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Overridden method; see base class (tf.keras.loss.Loss).

        Arguments:
            y_true: input tensor
            y_pred: output tensor of the autoencoder
            sample_weight: grid of the same dimension as target containing
                distances between each point and the nearest point with ligand
                density. This should be constructed by stacking n_channels
                copies of the 3D output of
                calculate_distances.calculate_distances on a new axis at
                position 1.

        Returns:
            4D tensor of the same shape as the input tensors. The mean squared
            error at each point is weighted by the distances according to the
            following expression in order to avoid dividing by zero errors:

                weighted_squared_error = squared_error*(1 / (distance**2 + 0.5)
        """
        proximities = 0.5 + 0.5 * (tf.math.tanh(4.0 - sample_weight))
        squared_difference = tf.math.squared_difference(y_true, y_pred)
        masked_sq_difference = tf.math.multiply(proximities, squared_difference)
        return tf.reduce_mean(masked_sq_difference, axis=self.reduction_axis)


class CompositeMse(tf.keras.losses.Loss, ABC):
    """Weighted mean squared error of nonzero-only and zero-only inputs."""

    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO,
                 name='composite_mse', reduction_axis=None):
        super().__init__(reduction=reduction, name=name)
        reduction_axis = reduction_axis if reduction_axis is not None else []
        self.reduction_axis = reduction_axis

    def call(self, y_true, y_pred):
        """Overridden method; see base class (tf.keras.loss.Loss).

        Finds the MSE between the autoencoder reconstruction and the nonzero
        entries of the input, the MSE between the reconstruction and the zero
        entries of the input and gives the weighted average of the two.

        Arguments:
            y_true: input tensor
            y_pred: output tensor of the autoencoder (reconstruction)

        Returns:
            Average weighted by:

                ratio/(1+ratio)*nonzero_mse + 1/(1+ratio)*zero_mse

            where nonzero_mse and zero_mse are the MSE for the nonzero and zero
            parts of target respectively."""

        nz_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        nz_mask_sum = tf.reduce_sum(nz_mask)
        nz_masked_difference = (y_true - y_pred) * nz_mask
        nz_mse = tf.square(nz_masked_difference)
        nz_mse_mean = tf.divide(tf.reduce_sum(nz_mse), nz_mask_sum)

        z_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        z_mask_sum = tf.reduce_sum(z_mask)
        z_masked_difference = (y_true - y_pred) * z_mask
        z_mse = tf.square(z_masked_difference)
        z_mse_mean = tf.divide(tf.reduce_sum(z_mse), z_mask_sum)

        frac = tf.divide(nz_mse_mean, tf.add(z_mse_mean, epsilon))
        frac = tf.divide(frac, 1. + frac)
        return tf.reduce_mean(tf.math.add(
            tf.math.multiply(frac, nz_mse),
            tf.math.multiply(1. - frac, z_mse)), axis=self.reduction_axis)


class BinaryCrossentropy(tf.keras.losses.Loss):

    def __init__(self, name='binary_crossentropy', reduction='none'):
        super().__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        bce = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
        bce += (1. - y_true) * tf.math.log(
            1. - y_pred + tf.keras.backend.epsilon())
        return -bce


def nonzero_mse(y_true, y_pred):
    """Mean squared error for non-zero values in the target matrix

    Finds the mean squared error for all parts of the input tensor which
    are not equal to zero.

    Arguments:
        y_true: input tensor
        y_pred: output tensor of the autoencoder

    Returns:
        Mean squared error for all non-zero entries in the target
    """
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    masked_difference = (y_true - y_pred) * mask
    return tf.square(masked_difference)


def zero_mse(y_true, y_pred):
    """Mean squared error for zero values in the target matrix

    Finds the mean squared error for all parts of the input tensor which
    are equal to zero.

    Arguments:
        y_true: input tensor
        y_pred: output tensor of the autoencoder

    Returns:
        Mean squared error for all zero entries in the target
    """
    mask = tf.cast(tf.equal(y_true, 0), tf.float32)
    masked_difference = (y_true - y_pred) * mask
    return tf.square(masked_difference)


def nonzero_mae(y_true, y_pred):
    """Mean absolute error loss function target values are not zero.

    Arguments:
        y_true: input tensor
        y_pred: output tensor of the autoencoder

    Returns:
        Tensor containing the mean absolute error between the target and
        the reconstruction where the mean is taken over values where
        the target is not zero.
        This can be NaN if there are no nonzero inputs.
    """
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    abs_diff = tf.abs(y_true - y_pred)
    masked_abs_diff = tf.math.multiply(abs_diff, mask)
    mask_sum = tf.reduce_sum(mask)
    return tf.reduce_sum(masked_abs_diff) / mask_sum


def zero_mae(y_true, y_pred):
    """Mean absolute error loss function target values are zero.

    Arguments:
        y_true: input tensor
        y_pred: output tensor of the autoencoder

    Returns:
        Tensor containing the mean absolute error between the target and
        the reconstruction where the mean is taken over values where
        the target is equal to zero.
        This can be NaN if there are no inputs equal to zero.
    """
    mask = tf.cast(tf.equal(y_true, 0), tf.float32)
    mask_sum = tf.reduce_sum(mask)
    abs_diff = tf.abs(y_true - y_pred)
    masked_abs_diff = tf.math.multiply(abs_diff, mask)
    return tf.reduce_sum(masked_abs_diff) / mask_sum


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
