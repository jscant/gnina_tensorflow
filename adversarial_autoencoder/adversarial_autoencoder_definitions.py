"""
Created on Sat Jun 20 12:30:08 2020

@author: scantleb
@brief: AutoEncoder class definition

Autoencoders learn a mapping from a high dimensional space to a lower
dimensional space, as well as the inverse.
"""
from abc import ABC
from collections import defaultdict
from pathlib import Path

import molgrid
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

from layers import dense_callable
from layers.layer_functions import generate_activation_layers
from utilities.gnina_functions import print_with_overwrite

crossentropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=False)

big = True


class Generator(tf.keras.Model):

    def __init__(self, input_shape, hidden_activation,
                 final_activation, latent_size):
        super().__init__(name='generator')
        self.inp_shape = input_shape
        encoding_activation_layer = next(generate_activation_layers(
            'encoding', hidden_activation, append_name_info=False))

        conv_activation = generate_activation_layers(
            'conv', hidden_activation, append_name_info=True)

        bn = lambda: layers.BatchNormalization(axis=1, epsilon=1.001e-5)

        conv_args = {'padding': 'same',
                     'data_format': 'channels_first',
                     'use_bias': False,
                     'kernel_initializer': tf.keras.initializers.HeNormal()}

        self.encoder_layers, self.decoder_layers = [], []
        self.encoder_layers.append(layers.Conv3D(128, 3, 2, **conv_args))
        self.encoder_layers.append(next(conv_activation))
        self.encoder_layers.append(bn())

        if big:
            self.encoder_layers.append(layers.Conv3D(256, 3, 2, **conv_args))
            self.encoder_layers.append(next(conv_activation))
            self.encoder_layers.append(bn())

            self.encoder_layers.append(layers.Conv3D(512, 3, 2, **conv_args))
            self.encoder_layers.append(next(conv_activation))
            self.encoder_layers.append(bn())

        final_shape_passthrough = np.zeros((1,) + input_shape, dtype='float32')
        for layer in self.encoder_layers:
            final_shape_passthrough = layer(final_shape_passthrough)
        self.final_shape = final_shape_passthrough.shape[1:]

        self.encoder_layers.append(layers.Flatten(data_format='channels_first'))
        self.encoder_layers.append(layers.Dense(latent_size))
        self.encoder_layers.append(encoding_activation_layer)

        self.decoder_layers.append(layers.Dense(np.prod(self.final_shape)))
        self.decoder_layers.append(next(conv_activation))
        self.decoder_layers.append(layers.Reshape(self.final_shape))

        if big:
            self.decoder_layers.append(
                layers.Conv3DTranspose(256, 3, 2, **conv_args))
            self.decoder_layers.append(next(conv_activation))
            self.decoder_layers.append(bn())

            self.decoder_layers.append(
                layers.Conv3DTranspose(128, 3, 2, **conv_args))
            self.decoder_layers.append(next(conv_activation))
            self.decoder_layers.append(bn())

        self.decoder_layers.append(
            layers.Conv3DTranspose(
                input_shape[0], 3, 2, name='reconstruction',
                activation=final_activation, **conv_args))

    def call(self, inputs, training=None, mask=None):
        latent_representation = self.encode(
            inputs, training=training)
        reconstruction = self.decode(
            latent_representation, training=training)
        return reconstruction

    def encode(self, inputs, training=None):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return x

    def decode(self, inputs, training=None):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(x, training=training)
        return x

    def summary(self, line_length=None, positions=None, print_fn=None):
        return self._model().summary(line_length, positions, print_fn)

    def _model(self):
        x = layers.Input(self.inp_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {'encoder_layers': self.encoder_layers,
             'decoder_layers': self.decoder_layers,
             'input_image': self.input_image,
             'final_shape': self.final_shape}
        )
        return config


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


class AdversarialAutoencoder:

    def __init__(self, ae_hidden_activation='swish',
                 ae_final_activation='sigmoid', ae_optimiser='adam',
                 ae_loss_fn='mse',
                 disc_optimiser='sgd', gen_opt_args=None,
                 disc_opt_args=None,
                 latent_size=500, binary_mask=False, ligmap=None, recmap=None,
                 data_root='~', train_types=None, dimension=23.5,
                 resolution=0.5, batch_size=16):

        def get_loss_fn(s):
            return {'composite_mse': CompositeMse,
                    'mse': SquaredError}[s]

        if disc_opt_args is None:
            disc_opt_args = {'lr': 0.01}

        self.batch_size = batch_size
        example_provider_kwargs = {
            'data_root': str(Path(data_root).expanduser()), 'balanced': False,
            'shuffle': True, 'cache_structs': False
        }
        if ligmap is None or recmap is None:
            # noinspection PyArgumentList
            self.e = molgrid.ExampleProvider(
                **example_provider_kwargs
            )
        else:
            rec_typer = molgrid.FileMappedGninaTyper(recmap)
            lig_typer = molgrid.FileMappedGninaTyper(ligmap)
            self.e = molgrid.ExampleProvider(
                rec_typer, lig_typer, **example_provider_kwargs)
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
        self.generator = Generator(input_shape, ae_hidden_activation,
                                   ae_final_activation, latent_size=latent_size)
        self.discriminator = Discriminator(input_shape)

        if ae_optimiser == 'adamw':
            self.gen_opt = tfa.optimizers.AdamW(**gen_opt_args)
        else:
            self.gen_opt = tf.keras.optimizers.get(ae_optimiser).__class__(
                **gen_opt_args
            )
        self.disc_opt = tf.keras.optimizers.get(disc_optimiser).__class__(
            **disc_opt_args
        )
        self.loss_fn = get_loss_fn(ae_loss_fn)(
            reduction=tf.keras.losses.Reduction.NONE)
        self.metrics = defaultdict(list)
        self.original_images = None

    @tf.function
    def _get_gradients(self, original_images):
        with tf.GradientTape() as discriminator_tape, \
                tf.GradientTape() as sim_tape, \
                tf.GradientTape() as recon_tape:
            reconstructions = self.generator(
                original_images, training=True)
            probabilities_real = self.discriminator(
                original_images, training=True)
            probabilities_fake = self.discriminator(
                reconstructions, training=True)

            similarity_loss = fake_quality(
                probabilities_fake)
            reconstruction_mse = self.loss_fn(
                original_images, reconstructions
            )

            discriminator_loss = discriminator_loss_fn(
                probabilities_real, probabilities_fake)

            nz_mae = nonzero_mae(original_images, reconstructions)

        gen_fake_grads = sim_tape.gradient(
            similarity_loss, self.generator.trainable_variables
        )
        gen_reconstruction_grads = recon_tape.gradient(
            reconstruction_mse, self.generator.trainable_variables
        )
        discriminator_grads = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables)

        self.gen_opt.apply_gradients(zip(
            gen_fake_grads, self.generator.trainable_variables
        ))

        self.gen_opt.apply_gradients(zip(
            gen_reconstruction_grads, self.generator.trainable_variables
        ))
        self.disc_opt.apply_gradients(zip(
            discriminator_grads, self.discriminator.trainable_variables))

        return (probabilities_real, probabilities_fake,
                similarity_loss, reconstruction_mse, discriminator_loss,
                nz_mae)

    def train_step(self):
        batch = self.e.next_batch(self.batch_size)
        self.gmaker.forward(batch, self.input_tensor, 0,
                            random_rotation=True)
        self.original_images = tf.convert_to_tensor(np.minimum(
            self.input_tensor.tonumpy(), 1.0).astype('float32'))
        return self._get_gradients(self.original_images)

    def train(self, iterations):
        loss_str, titles = None, None
        with open('adv_ae_losses_mae.txt', 'w') as f:
            f.write('')
        for self.iteration in range(self.iteration, iterations):

            (prob_real, prob_fake, similarity_loss, reconstruction_loss,
             disc_loss, nz_mae) = self.train_step()

            self.metrics['similarity_loss'].append(similarity_loss)
            self.metrics['reconstruction_loss'].append(
                np.mean(reconstruction_loss))
            self.metrics['discriminator_loss'].append(disc_loss)
            self.metrics['nonzero_mae'].append(nz_mae)
            self.metrics['real_probs'].append(np.mean(prob_real))
            self.metrics['fake_probs'].append(np.mean(prob_fake))
            if not self.iteration:
                print('\n')
            print_with_overwrite(
                ('Iteration: {0}\nGenerator loss (similarity): {1:0.4f}\n'
                 'Generator loss (reconstruction mse): {2:0.4f}\n'
                 'Discriminator loss: {3:0.4f}\n'
                 'Real probability: {4:0.7f}\n'
                 'Fake probability: {5:0.4f}\n'
                 'Nonzero mae: {6:0.4f}').format(
                    self.iteration + 1,
                    self.metrics['similarity_loss'][-1],
                    self.metrics['reconstruction_loss'][-1],
                    self.metrics['discriminator_loss'][-1],
                    self.metrics['real_probs'][-1],
                    self.metrics['fake_probs'][-1],
                    self.metrics['nonzero_mae'][-1]))

            if loss_str is None:
                titles = sorted(self.metrics.keys())
                loss_str = 'iteration ' + ' '.join(titles) + '\n'
                with open('adv_ae_losses.txt', 'w') as f:
                    f.write(loss_str)
                loss_str = ''

            n = 100
            if not (self.iteration + 1) % n:
                d = {t: np.array(self.metrics[t][-n:]) for t in titles}
                for iter in range(n):
                    loss_str += ' '.join(
                        [str(self.iteration + 2 - n + iter)] +
                        ['{0:0.4f}'.format(d[t][iter]) for t in titles])
                    loss_str += '\n'
                with open('adv_ae_losses_mae.txt', 'a') as f:
                    f.write(loss_str)
                loss_str = ''


def fake_quality(discriminator_predictions):
    similarity = crossentropy(tf.ones_like(discriminator_predictions),
                              discriminator_predictions)
    return similarity


def discriminator_loss_fn(real_predictions, fake_predictions):
    real_loss = crossentropy(tf.ones_like(real_predictions), real_predictions)
    fake_loss = crossentropy(tf.zeros_like(fake_predictions), fake_predictions)
    return 0.5 * (real_loss + fake_loss)


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
    mask = 1.0 - tf.cast(tf.equal(target, 0), tf.float32)
    mask_sum = tf.reduce_sum(mask, axis=None)
    abs_diff = tf.abs(target - reconstruction)
    masked_abs_diff = tf.math.multiply(abs_diff, mask)
    return float(tf.reduce_sum(masked_abs_diff, axis=None) / mask_sum)


class SquaredError(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.cast(y_pred, y_true.dtype)
        return tf.square(y_true - y_pred)


class CompositeMse(tf.keras.losses.Loss, ABC):
    """Weighted mean squared error of nonzero-only and zero-only inputs."""

    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name='composite_mse'):
        super().__init__(reduction=reduction, name=name)

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
            parts of target respectively.
        """
        nz_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        nz_masked_difference = (y_true - y_pred) * nz_mask
        nz_mse = tf.abs(nz_masked_difference)
        nz_mask_sum = tf.reduce_sum(nz_mask)

        z_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        z_masked_difference = (y_true - y_pred) * z_mask
        z_mse = tf.abs(z_masked_difference)
        z_mask_sum = tf.reduce_sum(z_mask)

        epsilon = tf.constant(1e-6, dtype=tf.float32)

        nz_mean = nz_mse / (nz_mask_sum + epsilon)
        z_mean = z_mse / (z_mask_sum + epsilon)
        frac = nz_mean / (z_mean + epsilon)
        frac = tf.divide(frac, 1. + frac)
        return tf.math.add(
            tf.math.multiply(frac, nz_mse),
            tf.math.multiply(1. - frac, z_mse))


a = AdversarialAutoencoder(train_types='data/dude_translated.types',
                           ligmap='data/gnina35.ligmap',
                           recmap='data/gnina35.recmap',
                           data_root='data', resolution=1.0, dimension=23.0,
                           disc_optimiser='sgd',
                           ae_optimiser='adamw',
                           disc_opt_args={'lr': 0.001, 'momentum': 0.9,
                                          'nesterov': True},
                           gen_opt_args={'lr': 0.0005, 'weight_decay': 1e-4},
                           latent_size=2000,
                           batch_size=16,
                           ae_hidden_activation='prelu',
                           ae_loss_fn='composite_mse')

a.generator.summary()
tf.keras.utils.plot_model(
    a.discriminator._model(), 'discriminator.png', show_layer_names=True,
    show_shapes=True)
tf.keras.utils.plot_model(
    a.generator._model(), 'generator.png', show_layer_names=True,
    show_shapes=True)
a.train(300000)
