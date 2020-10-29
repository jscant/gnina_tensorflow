"""
Created on Sat Jun 20 12:30:08 2020

@author: scantleb
@brief: AutoEncoder class definition

Autoencoders learn a mapping from a high dimensional space to a lower
dimensional space, as well as the inverse.
"""

import os
from pathlib import Path

import molgrid
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from layers import dense_callable
from layers.layer_functions import generate_activation_layers
from utilities.gnina_functions import print_with_overwrite

crossentropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True)


class Generator(tf.keras.Model):

    def __init__(self, input_shape, hidden_activation,
                 final_activation, latent_size):
        super().__init__(name='generator')
        encoding_activation_layer = next(generate_activation_layers(
            'encoding', hidden_activation, append_name_info=False))

        conv_activation = generate_activation_layers(
            'conv', hidden_activation, append_name_info=True)

        bn = lambda: layers.BatchNormalization(axis=1, epsilon=1.001e-5)

        self.input_image = layers.Input(shape=input_shape, dtype=tf.float32,
                                        name='input_image')

        conv_args = {'padding': 'same',
                     'data_format': 'channels_first',
                     'use_bias': False}

        self.encoder_layers, self.decoder_layers = [], []
        self.encoder_layers.append(layers.Conv3D(128, 3, 2, **conv_args))
        self.encoder_layers.append(next(conv_activation))
        self.encoder_layers.append(bn())

        self.encoder_layers.append(layers.Conv3D(256, 3, 2, **conv_args))
        self.encoder_layers.append(next(conv_activation))
        self.encoder_layers.append(bn())

        self.encoder_layers.append(layers.Conv3D(512, 3, 2, **conv_args))
        self.encoder_layers.append(next(conv_activation))
        self.encoder_layers.append(bn())

        final_shape_passthrough = self.encoder_layers[0](self.input_image)
        for layer in self.encoder_layers[1:]:
            final_shape_passthrough = layer(final_shape_passthrough)
        self.final_shape = final_shape_passthrough.shape[1:]

        self.encoder_layers.append(layers.Flatten(data_format='channels_first'))
        self.encoder_layers.append(layers.Dense(latent_size))
        self.encoder_layers.append(encoding_activation_layer)

        self.decoder_layers.append(layers.Dense(np.prod(self.final_shape)))
        self.decoder_layers.append(next(conv_activation))
        self.decoder_layers.append(layers.Reshape(self.final_shape))

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
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x

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

        self.disc_layers = []
        self.disc_layers.append(layers.Flatten(data_format='channels_first'))
        self.disc_layers.append(layers.Dense(1000, activation='relu'))
        self.disc_layers.append(layers.Dense(
            2, activation='softmax', name='disc_probabilities'))

        self.global_max = layers.GlobalMaxPooling3D(
            data_format='channels_first')

        self.probabilities = layers.Dense(
            2, activation='softmax', name='probabilities')

    def call(self, inputs, training=None, mask=None):
        x = self.mp_1(inputs)
        x = self.conv_1(x)
        for idx, layer in enumerate(self.dense_layers):
            x = layer(x)
            if idx == 1 and self.disc:
                for disc_layer in self.disc_layers:
                    x = disc_layer(x)
                return x

        x = self.global_max(x)
        return self.probabilities(x)

    def get_config(self):
        pass


class AdversarialAutoencoder:

    def __init__(self, ae_hidden_activation='swish',
                 ae_final_activation='sigmoid', ae_optimiser='adamax',
                 disc_optimiser='sgd', gen_opt_args=None,
                 disc_opt_args=None,
                 latent_size=500, binary_mask=False, ligmap=None, recmap=None,
                 data_root='~', train_types=None, dimension=23.5,
                 resolution=0.5, batch_size=16):

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
                                   ae_final_activation, latent_size)
        self.discriminator = Discriminator(input_shape)

        self.gen_opt = tf.keras.optimizers.get(ae_optimiser).__class__(
            **gen_opt_args
        )
        self.disc_opt = tf.keras.optimizers.get(disc_optimiser).__class__(
            **disc_opt_args
        )
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def _train_step(self):
        batch = self.e.next_batch(self.batch_size)
        self.gmaker.forward(batch, self.input_tensor, 0,
                            random_rotation=True)

        original_images = tf.convert_to_tensor(np.minimum(
            self.input_tensor.tonumpy(), 1.0).astype('float32'))

        # with tf.GradientTape() as generator_tape, \
        #        tf.GradientTape() as discriminator_tape:
        with tf.GradientTape() as discriminator_tape, \
                tf.GradientTape() as sim_tape, \
                tf.GradientTape() as recon_tape:
            reconstructions = self.generator(
                original_images, training=True)
            probabilities_real = self.discriminator(
                original_images, training=True)
            probabilities_fake = self.discriminator(
                reconstructions, training=True)
            # generator_loss = generator_loss_fn(
            #    probabilities_fake, original_images, reconstructions)
            similarity_loss = fake_quality(
                probabilities_fake
            )
            reconstruction_mse = self.mse(
                original_images, reconstructions
            )
            discriminator_loss = discriminator_loss_fn(
                probabilities_real, probabilities_fake)

            nz_mae = nonzero_mae(original_images, reconstructions)

        # generator_grads = generator_tape.gradient(
        #    generator_loss, self.generator.trainable_variables)

        gen_fake_grads = sim_tape.gradient(
            similarity_loss, self.generator.trainable_variables
        )
        gen_reconstruction_grads = recon_tape.gradient(
            reconstruction_mse, self.generator.trainable_variables
        )
        discriminator_grads = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables)

        # self.gen_opt.apply_gradients(zip(
        #    generator_grads, self.generator.trainable_variables))

        self.gen_opt.apply_gradients(zip(
            gen_fake_grads, self.generator.trainable_variables
        ))
        self.gen_opt.apply_gradients(zip(
            gen_reconstruction_grads, self.generator.trainable_variables
        ))
        self.disc_opt.apply_gradients(zip(
            discriminator_grads, self.discriminator.trainable_variables))

        return (probabilities_real[:, 1], probabilities_fake[:, 1],
                similarity_loss, reconstruction_mse, discriminator_loss,
                nz_mae)

    def train(self, iterations):
        sim_losses, recon_losses, disc_losses, nz_maes = [], [], [], []
        real_probs, fake_probs = [], []
        self.discriminator.disc = True
        tf.keras.utils.plot_model(
            self.discriminator, Path('~/projects/junk/model.png').expanduser(),
            show_layer_names=True, show_shapes=True)
        for self.iteration in range(self.iteration, iterations):
            (prob_real, prob_fake, similarity_loss, reconstruction_loss,
             disc_loss, nz_mae) = self._train_step()
            sim_losses.append(similarity_loss)
            recon_losses.append(reconstruction_loss)
            disc_losses.append(disc_loss)
            nz_maes.append(nz_mae)
            real_probs.append(prob_real)
            fake_probs.append(prob_fake)
            if not self.iteration:
                print('\n')
            print_with_overwrite(
                ('Iteration: {0}\nGenerator loss (similarity): {1:0.4f}\n'
                 'Generator loss (reconstruction mse): {2:0.4f}\n'
                 'Discriminator loss: {3:0.4f}\n'
                 'Real probability: {4:0.4f}\n'
                 'Fake probability: {5:0.4f}\n'
                 'Nonzero mae: {6:0.4f}').format(
                    self.iteration + 1,
                    similarity_loss,
                    np.mean(reconstruction_loss),
                    disc_loss,
                    np.mean(prob_real), np.mean(prob_fake),
                    nz_mae))

            if not (self.iteration + 1) % 100:
                with open('adv_ae_losses.txt', 'w') as f:
                    s = 'similarity reconstruction_mse disc\n'
                    s += '\n'.join(
                        ['{0} {1} {2}'.format(i, j, k) for i, j, k in
                         zip(sim_losses, recon_losses, disc_losses)])
                    s += '\n'
                    f.write(s)

                with open('probs.txt', 'w') as f:
                    s = 'real fake\n'
                    s += '\n'.join(
                        ['{0} {1}'.format(np.mean(i), np.mean(j)) for i, j in
                         zip(real_probs, fake_probs)])
                    s += '\n'
                    f.write(s)
                #os.system(
                #    'python3 {}'.format(
                #        Path('~/projects/junk/plot_probs.py').expanduser()))


def generator_loss_fn(discriminator_predictions, input_image, reconstruction):
    mse = tf.reduce_mean(tf.square(input_image - reconstruction))
    feasibility = crossentropy(tf.ones_like(discriminator_predictions),
                               discriminator_predictions)

    ratio = mse / feasibility
    return (ratio * feasibility) + mse


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
    mask_sum = tf.reduce_sum(mask)
    abs_diff = tf.abs(target - reconstruction)
    masked_abs_diff = tf.math.multiply(abs_diff, mask)
    return tf.reduce_sum(masked_abs_diff) / mask_sum


a = AdversarialAutoencoder(train_types='data/dude_translated.types',
                           data_root='data', resolution=1.0, dimension=23.0,
                           disc_optimiser='sgd',
                           disc_opt_args={'lr': 0.001, 'momentum': 0.9,
                                          'nesterov': True},
                           gen_opt_args={'lr': 0.001},
                           latent_size=2000)

a.train(25000)
