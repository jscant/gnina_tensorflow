"""
Created on Sat Jun 20 12:30:08 2020

@author: scantleb
@brief: AutoEncoder class definition

Autoencoders learn a mapping from a high dimensional space to a lower
dimensional space, as well as the inverse.
"""

import argparse
import tensorflow as tf
import numpy as np

from classifier.model_definitions import tf_transition_block,\
    tf_inverse_transition_block, tf_dense_block
from functools import reduce
from operator import mul
from pathlib import Path
from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, \
    MaxPooling3D, Reshape, Conv3DTranspose, UpSampling3D, BatchNormalization


def long_sigmoid(x):
    """2.3 times the sigmoid function."""
    return tf.math.multiply(2.3, tf.nn.sigmoid(x))


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
    mask = 1. - tf.cast(tf.equal(target, 0), tf.float32)
    masked_difference = (target - reconstruction) * mask
    return tf.reduce_mean(tf.square(masked_difference))


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
    return tf.reduce_mean(tf.square(masked_difference))


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
    frac = tf.divide(ratio, 1+ratio)
    return tf.math.add(
        tf.math.multiply(frac, nonzero_mse(target, reconstruction)),
        tf.math.multiply(1-frac, zero_mse(target, reconstruction)))


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
    masked_diff = (target - reconstruction) * mask
    abs_diff = tf.abs(masked_diff)
    return tf.divide(tf.reduce_sum(abs_diff), tf.reduce_sum(mask))


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
    mask = 1 - tf.cast(tf.equal(target, 0), tf.float32)
    mask_sum = tf.reduce_sum(mask)
    masked_diff = (target - reconstruction) * mask
    abs_diff = tf.abs(masked_diff)
    return tf.divide(tf.reduce_sum(abs_diff), mask_sum)


def unbalanced_loss(target, reconstruction):
    """Loss function which more heavily penalises loss on nonzero terms.

    Only use for positive labels.

    Arguments:
        y_true: tensor containing true labels for model inputs
        y_pred: tensor containing output(s) of model

    Returns:
        Error weighted strongly in favour of penalising errors on parts
        of y_true which are above zero.
    """
    term_1 = tf.pow(target - reconstruction, 4)
    term_2 = tf.pow(target - reconstruction, 2)
    loss = target*term_1 + tf.multiply(
        tf.constant(0.01), tf.multiply(1 - target, term_2))
    return tf.reduce_mean(loss)


def approx_heaviside(x):
    """Activation function: continuous approximation to Heaviside fn.

    Arguments:
        x: Tensor with pre-activation values.

    Returns:
        Tensor containing activations, where activations at or below zero
        are (approx.) zero, and activations above zero are (approx) one.
    """
    return 0.5 + 0.5*tf.tanh(x*10.)


class AutoEncoderBase(tf.keras.Model):
    """Virtual parent class for autoencoders."""

    def __init__(self, optimiser, loss, opt_args):
        """Initialisation of base class.

        Arguments:
            optimiser: any keras optimisation class
            loss: any keras loss fuction (or string reference), or or
                'unbalanced'/'composite_mse' (custom weighted loss functions)
            opt_args: arguments for the keras optimiser (see keras
                documentation)
        """

        optimisers = {
            'sgd': tf.keras.optimizers.SGD,
            'adadelta': tf.keras.optimizers.Adadelta,
            'adagrad': tf.keras.optimizers.Adagrad,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'adamax': tf.keras.optimizers.Adamax,
            'adam': tf.keras.optimizers.Adam
        }

        # If optimiser is a string, turn it into a keras optimiser object
        if isinstance(optimiser, str):
            optimiser = optimisers.get(optimiser, tf.keras.optimizers.SGD)

        inputs = [self.input_image]
        if loss == 'composite_mse':
            self.frac = Input(shape=(1,), dtype=tf.float32, name='frac')
            inputs.append(self.frac)

        super(AutoEncoderBase, self).__init__(
            inputs=inputs,
            outputs=[self.reconstruction, self.encoding]
        )

        metrics = {'reconstruction': [mae, nonzero_mae, zero_mae,
                                      zero_mse, nonzero_mse]}
        if loss == 'composite_mse':
            self.add_loss(composite_mse(
                self.input_image, self.reconstruction, self.frac))
            self.compile(
                optimizer=optimiser(**opt_args),
                metrics=metrics
            )
        else:
            if loss == 'unbalanced':
                loss = unbalanced_loss
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

    def _define_activations(self):
        """Returns dict from strings to final layer activations objects."""

        return {'sigmoid': 'sigmoid',
                'relu': 'relu',
                'heaviside': approx_heaviside}


class AutoEncoder(AutoEncoderBase):

    def __init__(self,
                 dims,
                 encoding_size=10,
                 optimiser='sgd',
                 loss='mse',
                 final_activation='sigmoid',
                 **opt_args):
        """Setup for autoencoder architecture.

        Arguments:
            dims: dimentionality of inputs
            encoding_size: size of bottleneck
            optimiser: keras optimiser (string)
            loss: loss function (string or keras loss object)
            final_activation: activation function for final layer
            opt_args: arguments for the keras optimiser (see keras
                documentation)
        """

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
    """Convolutional autoencoder with Dense connectivity."""

    def __init__(self, dims, encoding_size=10,
                 optimiser=tf.keras.optimizers.SGD,
                 loss='mse',
                 final_activation='sigmoid',
                 **opt_args):
        """Setup for autoencoder architecture.

        Arguments:
            dims: dimentionality of inputs
            encoding_size: size of bottleneck
            optimiser: keras optimiser (string)
            loss: loss function (string or keras loss object)
            final_activation: activation function for final layer
            opt_args: arguments for the keras optimiser (see keras
                documentation)"""

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
        """Setup for single layer autoencoder architecture.

        Arguments:
            dims: dimentionality of inputs
            encoding_size: size of bottleneck
            optimiser: keras optimiser (string)
            loss: loss function (string or keras loss object)
            final_activation: activation function for final layer
            opt_args: arguments for the keras optimiser (see keras
                documentation)
        """

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


class LoadConfigTrain(argparse.Action):
    """Class for loading argparse arguments from a config file."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Overloaded function; See parent class."""

        if values is None:
            return
        config = Path(values).parents[1] / 'config'
        if not config.exists():
            print("No config file found in experiment's base directory ({})".format(
                    config))
            print('Only specified command line args will be used.')
            namespace.load_model = values
            return
        args = ''
        with open(config, 'r') as f:
            for line in f.readlines():
                chunks = line.split()
                if not len(chunks): continue
                if chunks[0] not in ['load_model',
                                     'absolute_save_path',
                                     'use_cpu',
                                     'binary_mask',
                                     'save_encodings']:
                    args += '--{0} {1}\n'.format(*chunks)
                else:  # store_true args present a problem, loaded manually
                    if chunks[1] == 'True':
                        args += '--{0}\n'.format(chunks[0])
        parser.parse_args(args.split(), namespace)

        # args.load_model is always None if we do not do this, even when
        # it is specified using --load_model.
        namespace.load_model = values
        
        
class LoadConfigTest(argparse.Action):
    """Class for loading argparse arguments from a config file."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Overloaded function; See parent class."""

        if values is None:
            return

        config = Path(values).parents[1] / 'config'
        if not config.exists():
            print("No config file found in experiment's base directory ({})".format(
                    config))
            print('Only specified command line args will be used.')
            namespace.load_model = values
            return
        args = ''
        namespace.binary_mask = False
        namespace.use_cpu = False
        with open(config, 'r') as f:
            for line in f.readlines():
                chunks = line.split()
                if not len(chunks): continue
                if chunks[0] in ['data_root', 'save_path']:
                    args += '--{0} {1}\n'.format(*chunks)
                elif chunks[0] in ['dimension', 'resolution']:
                    setattr(namespace, chunks[0], float(chunks[1]))
                else:  # store_true args present a problem, loaded manually
                    if chunks[0] == 'binary_mask':
                        if len(chunks) == 1 or chunks[1] == 'True':
                            namespace.binary_mask = True
                    if chunks[0] == 'use_cpu':
                        if len(chunks) == 1 or chunks[1] == 'True':
                            namespace.use_cpu = True

        parser.parse_args(args.split(), namespace)

        # args.load_model is always None if we do not do this, even when
        # it is specified using --load_model.
        namespace.load_model = values


def pickup(path):
    """Loads saved autoencoder.

    Arguments:
        path: location of saved weights and architecture

    Returns:
        AutoEncoderBase-derived object initialised with weights from saved
        checkpoint.
    """

    ae = tf.keras.models.load_model(
        path,
        custom_objects={
            'zero_mse': zero_mse,
            'nonzero_mse': nonzero_mse,
            'composite_mse': composite_mse,
            'nonzero_mae': nonzero_mae,
            'zero_mae': zero_mae,
            'approx_heaviside': approx_heaviside,
            'unbalanced_loss': unbalanced_loss,
        }
    )

    # Bug with add_loss puts empty dict at the end of model._layers which
    # interferes with some functionality (such as
    # tf.keras.utils.plot_model)
    ae._layers = [layer for layer in ae._layers if isinstance(
        layer, tf.keras.layers.Layer)]
    return ae


def parse_command_line_args(test_or_train='train'):
    """Parse command line args and return as dict.

    Returns a dictionary containing all args, default or otherwise; if 'pickup'
    is specified, as many args as are contained in the config file for that
    (partially) trained model are loaded, otherwise defaults are given.
    Command line args override args found in the config of model found in
    'pickup' directory.
    """
    
    parser = argparse.ArgumentParser()
    
    if test_or_train == 'train':
        parser.add_argument(
            'load_model', type=str, action=LoadConfigTrain,
            nargs='?', help=
            """Load saved keras model. If specified, this should be the 
            directory containing the assets of a saved autoencoder. 
            If specified, the options are loaded from the config file saved
            when the original model was trained; any options specified in the 
            command line will override the options loaded from the config file.
            """)
        parser.add_argument("--train", '-t', type=str, required=False)
        parser.add_argument('--encoding_size', '-e', type=int, required=False,
                            default=50)
        parser.add_argument('--iterations', '-i', type=int, required=False)
        parser.add_argument(
            '--save_interval', type=int, required=False, default=10000)
        parser.add_argument(
            '--model', '-m', type=str, required=False, default='single',
            help='Model architecture; one of single (SingleLayerAutoencoder' +
            '), dense (DenseAutoEncodcer) or auto (AutoEncoder)')
        parser.add_argument(
            '--optimiser', '-o', type=str, required=False, default='sgd')
        parser.add_argument(
            '--learning_rate', '-l', type=float, required=False)
        parser.add_argument(
            '--momentum', type=float, required=False, default=0.0)
        parser.add_argument(
            '--loss', type=str, required=False, default='mse')
        parser.add_argument(
            '--final_activation', type=str, required=False, default='sigmoid')
        parser.add_argument('--binary_mask', action='store_true')
        parser.add_argument(
            '--dimension', type=float, required=False, default=18.0)
        parser.add_argument(
            '--resolution', type=float, required=False, default=1.0)
        parser.add_argument(
        '--save_encodings', action='store_true')
    else:
        parser.add_argument(
            '--load_model', type=str, action=LoadConfigTest, help=
            """Load saved keras model. If specified, this should be the 
            directory containing the assets of a saved autoencoder. 
            If specified, the options are loaded from the config file saved
            when the original model was trained; any options specified in the 
            command line will override the options loaded from the config file.
            """)
        parser.add_argument("--test", '-t', type=str, required=False)
        
    parser.add_argument("--data_root", '-r', type=str, required=False,
                        default='')
    parser.add_argument(
        '--batch_size', '-b', type=int, required=False, default=16)
    parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.')
    parser.add_argument(
        '--use_cpu', '-g', action='store_true')
    args = parser.parse_args()

    autoencoder = None
    if args.load_model is not None:  # Load a model
        autoencoder = pickup(args.load_model)
    elif test_or_train == 'test':
        raise RuntimeError(
            'Please specify a model to use to calculate encodings.')

    return autoencoder, args
