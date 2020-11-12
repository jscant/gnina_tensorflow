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

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, \
    MaxPooling3D, GlobalMaxPooling3D

from layers import dense, dense_callable


class DenseFS(tf.keras.Model):

    def __init__(self, input_shape):
        super().__init__(name='DenseFS')
        self._input_shape = input_shape
        self.dense_layers = []

        self.input_image = layers.Input(
            shape=input_shape, dtype=tf.float32, name='input_image')

        self.mp_1 = layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same',
            data_format='channels_first')

        self.conv_1 = layers.Conv3D(
            32, 3, activation='relu', padding='same', use_bias=False,
            data_format='channels_first')

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

        self.probabilities = layers.Dense(
            1, activation='sigmoid', name='probabilities')

        self.opt = tf.keras.optimizers.SGD(
            learning_rate=0.01, momentum=0.9)
        self.loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False)

    def call(self, inputs, training=None, mask=None):
        x = self.mp_1(inputs, training=training)
        x = self.conv_1(x, training=training)
        for layer in self.dense_layers:
            x = layer(x, training=training)
        x = self.global_max(x, training=training)
        return self.probabilities(x, training=training)

    def summary(self, line_length=None, positions=None, print_fn=None):
        return self._model().summary(line_length, positions, print_fn)

    def _model(self):
        x = layers.Input(self._input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def plot(self, to_file='model.png', show_shapes=False,
             show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96):
        """See tf.keras.plot_model() documentation for args.

        Saves a graphical representation of model to disk."""
        return tf.keras.utils.plot_model(
            self._model(), to_file=to_file, show_shapes=show_shapes,
            show_layer_names=show_layer_names, rankdir=rankdir,
            expand_nested=expand_nested, dpi=dpi)

    def get_config(self):
        return super().get_config()

    @tf.function
    def train_step(self, input_numpy, labels):
        """Perform backpropagation on a single batch.

        The tf.function decoration causes this function to be executed in graph
        mode. This has the effect of a massive increase in the speed of training
        except when using the distance_mse loss option which is significantly
        slowed down.

        Arguments:
            labels: binary labels indicating binding activity

        Returns:
            The binary crossentropy loss for the batch.
        """
        # batch = self.e.next_batch(self.batch_size)
        # self.gmaker.forward(batch, self.input_tensor, 0,
        #                    random_rotation=True)
        # input_numpy = self.input_tensor.tonumpy()
        # original_images = tf.convert_to_tensor(np.minimum(1.0, input_numpy))

        with tf.GradientTape() as tape:
            probabilities = self.call(input_numpy, training=True)
            loss = self.loss(labels, probabilities)
            grads = tape.gradient(
                loss, self.trainable_variables
            )
            self.opt.apply_gradients(zip(
                grads, self.trainable_variables
            ))
        return float(tf.reduce_mean(loss))


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
    input_layer = Input(shape=dims, dtype=tf.float32, name='input_image')

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
    output_layer = Dense(
        2, activation='softmax', name='probabilities')(representation)

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
