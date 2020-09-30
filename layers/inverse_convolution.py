#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:55:13 2020
@author: scantleb
@brief: Layer definition for artifact-free upsample and convolution operation.

The Conv3DTranspose must be used with great care to avoid checkerboard artifacts
on reconstructed images (for example, the stride must be devisible by the
kernel size). Using this 'deconvolution' has thus fallen out of favour, being
replaced by upsampling (using nearest neighbour interpolation) followed by
a convolution with stride 1.
"""

from tensorflow.keras import layers


class InverseConvolution3D(layers.Layer):
    """Convolution with upsampling."""

    def __init__(self, filters, kernel_size, scale_factor, **kwargs):
        """Constructor for layer.

        Arguments:
            filters: number of filters in output of layer
            kernel_size: int or tuple of 3 ints, size of kernel to apply to
                inputs
            scale_factor: factor to multiply spatial dimensions of inputs by
            kwargs: other arguments for the 3D convolution (see documentation
                for layers.Conv3D)
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.filters = filters
        self.kernel_size = kernel_size

        self.upsample = layers.UpSampling3D(
            self.scale_factor,
            data_format=kwargs.get('data_format', 'channels_first')
        )
        self.conv = layers.Conv3D(self.filters, self.kernel_size, 1, **kwargs)

    def __call__(self, inputs):
        """Overloaded method; see base class (layers.Layer).

        Performs an upsample operation before a convolution.
        """
        x = self.upsample(inputs)
        x = self.conv(x)
        return x

    def get_config(self):
        """Overloaded method; see base class (layers.Layer)."""
        config = super().get_config()
        config.update(
            {
                'scale_factor': self.scale_factor,
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'upsample': self.upsample,
                'conv': self.conv
            }
        )
        return config
