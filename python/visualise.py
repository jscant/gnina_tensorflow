#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:32:56 2020

@author: scantleb
"""

import numpy as np
from matplotlib import pyplot as plt

test_input = np.random.rand(28, 48, 48, 48)


def scatter(img, channel):
    img = img[channel, :, :, :].squeeze()
    xlin, ylin, zlin = np.arange(-11.5, 12.5, 0.5), np.arange(-11.5,
                                                              12.5, 0.5), np.arange(-11.5, 12.5, 0.5)
    x, y, z = np.meshgrid(xlin, ylin, zlin)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=img)


scatter(test_input, channel=1)
