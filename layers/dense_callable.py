import tensorflow as tf
from tensorflow.keras import layers

from layers.layer_functions import generate_activation_layers


class TransitionBlock(layers.Layer):

    def __init__(self, input_shape, reduction, name, activation='relu',
                 final=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.final = final
        conv_initialiser = tf.keras.initializers.HeNormal()
        self.act_0 = next(generate_activation_layers(
            '{0}_{1}'.format(name, activation), activation))
        bn_axis = 1
        self.bn = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999),
            name=name + '_bn')

        if self.final:  # No conv or maxpool, will global pool after final TB
            return

        self.conv = layers.Conv3D(input_shape[bn_axis] * reduction, 1,
                                  data_format='channels_first', use_bias=False,
                                  kernel_initializer=conv_initialiser,
                                  name=name + '_{}'.format(activation))
        self.max_pool = layers.MaxPooling3D(2, strides=2, name=name + '_pool',
                                            data_format='channels_first')

    def call(self, inputs, **kwargs):
        x = self.bn(inputs)
        if self.final:
            return x
        x = self.conv(x)
        x = self.act_0(x)
        return self.max_pool(x)


class DenseBlock(layers.Layer):

    def __init__(self, blocks, name, activation='relu', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv_blocks = []
        for i in range(blocks):
            self.conv_blocks.append(ConvBlock(
                16, '{0}_block_{1}'.format(name, i + 1), activation=activation))

    def call(self, inputs, **kwargs):
        x = inputs
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x


class ConvBlock(layers.Layer):

    def __init__(self, growth_rate, name, activation='relu', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv_initialiser = tf.keras.initializers.HeNormal()
        self.act_0 = next(generate_activation_layers(
            '{0}_{1}'.format(name, activation), activation))
        bn_axis = 1
        self.bn = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999),
            name=name + '_0_bn')
        self.conv = layers.Conv3D(
            growth_rate, 3, use_bias=False, padding='same',
            name=name + '_0_{}'.format(activation),
            kernel_initializer=self.conv_initialiser,
            data_format='channels_first')
        self.concat = layers.Concatenate(axis=bn_axis, name=name + '_concat')

    def call(self, inputs, **kwargs):
        x = self.bn(inputs)
        x = self.conv(x)
        x = self.act_0(x)
        return self.concat([inputs, x])


class InverseTransitionBlock(tf.keras.layers.Layer):

    def __init__(self, input_shape, reduction, name, activation='relu',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        conv_initialiser = tf.keras.initializers.HeNormal()
        self._input_shape = input_shape
        self.act_0 = next(generate_activation_layers(
            '{0}_{1}'.format(name, activation), activation))
        self.bn = layers.BatchNormalization(
            axis=1, epsilon=1.001e-5,
            moving_mean_initializer=tf.constant_initializer(0.999),
            name=name + '_bn')
        self.conv = layers.Conv3D(
            int(reduction * self._input_shape[1]), 1,
            data_format='channels_first', use_bias=False,
            kernel_initializer=conv_initialiser,
            name=name + '_{}'.format(activation))
        self.upsample = layers.UpSampling3D(
            2, name=name + '_upsample', data_format='channels_first')

    def call(self, inputs, **kwargs):
        x = self.bn(inputs)
        x = self.conv(x)
        x = self.act_0(x)
        x = self.upsample(x)
        return x
