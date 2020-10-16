"""
Created on Fri Jul 10 14:47:44 2020

@author: scantleb
@brief: Utility functions for use in various machine learning models
"""

import math
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


class Timer:
    """Simple timer class.

    To time a block of code, wrap it like so:

        with Timer() as t:
            <some_code>
        total_time = t.interval

    The time taken for the code to execute is stored in t.interval.
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def wipe_directory(directory):
    """Recursively removes all items in a directory, then the directory itself.

    Arguments:
        directory: Location of directory to wipe (str or PosixPath)
    """
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            wipe_directory(item)
        else:
            item.unlink()
    directory.rmdir()


def get_dims(dimension, resolution, ligmap, recmap):
    """Get the dimensions for a given dimension, resolution and channel setting.

    Arguments:
        dimension: length of side of cube in which ligand is situated, in
            Angstroms
        resolution: resolution of voxelisation of cube in which ligand is
            situated, in Angstroms
        ligmap: text file with ligand channel setup
        recmap: text file with receptor channel setup

    Returns:
        Tuple containing dimensions of gnina input
    """
    channels = 0
    for fname in [ligmap, recmap]:
        if fname is None:
            c = 14
        else:
            with open(fname, 'r') as f:
                c = sum([1 for line in f.readlines() if len(line)])
        channels += c
    length = int((dimension + 1) // resolution)
    return channels, length, length, length


def format_time(t):
    """Returns string continaing time in hh:mm:ss format.

    Arguments:
        t: time in seconds
        
    Raises:
        ValueError if t < 0
    """
    if t < 0:
        raise ValueError('Time must be positive.')

    t = int(math.floor(t))
    h = t // 3600
    m = (t - (h * 3600)) // 60
    s = t - ((h * 3600) + (m * 60))
    return '{0:02d}:{1:02d}:{2:02d}'.format(h, m, s)


def print_with_overwrite(s):
    """Prints to console, but overwrites previous output, rather than creating
    a newline.
    
    Arguments:
        s: string (possibly with multiple lines) to print
    """
    ERASE = '\x1b[2K'
    UP_ONE = '\x1b[1A'
    lines = s.split('\n')
    n_lines = len(lines)
    console_width = shutil.get_terminal_size((0, 20)).columns
    for idx in range(n_lines):
        lines[idx] += ' ' * max(0, console_width - len(lines[idx]))
    print((ERASE + UP_ONE) * (n_lines - 1) + s, end='\r', flush=True)


def get_test_info(test_file):
    """Obtains information about gninatypes file.

    Arguments:
        test_file: text file containing labels and paths to gninatypes files

    Returns:
        dictionary containing tuples with the format:
            {index : (receptor_path, ligand_path)}
            where index is the line number, receptor_path is the path to the
            receptor gninatype and ligand_path is the path to the ligand
            gninatype.
    """
    paths = {}
    with open(test_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            chunks = line.strip().split()
            paths[idx] = (chunks[-2], chunks[-1])
    return paths, len(paths)


def process_batch(model, example_provider, gmaker, input_tensor,
                  labels_tensor=None, train=True, autoencoder=None):
    """Feeds forward and backpropagates (if train==True) batch of examples.

    Arguments:
        model: compiled tensorflow model
        example_provider: molgrid.ExampleProvider object populated with a
            types file
        gmaker: molgrid.GridMaker object
        input_tensor: molgrid.MGrid<x>f object, where <x> is the dimentions
            of the input (including a dimention for batch size)
        labels_tensor: molgrid.MGrid1f object, for storing true labels. If
            labels_tensor is None and train is False, , return value will be
            a vector of predictions.
        train: are we training (performing backpropagation)
        autoencoder: trained autoencoder model to feed inputs through; the
            reconstruction is used as the input to the main model

    Returns:
        if labels_tensor is None and train is False: numpy.ndarray of
            predictions
        if labels_tensor is specified and train is False: tuple containing
            numpy.ndarray of labels and numpy.ndarray of predictions
        if labels_tensor is specified and train is True: float containing
            the mean loss over the batch (usually cross-entropy)

    Raises:
        RuntimeError: if labels_tensor is None and train is True
    """
    if train and labels_tensor is None:
        raise RuntimeError('Labels must be provided for backpropagation ',
                           'if train == True')

    batch_size = input_tensor.shape[0]
    batch = example_provider.next_batch(batch_size)
    gmaker.forward(batch, input_tensor, 0, random_rotation=train)

    input_numpy = input_tensor.tonumpy()
    tf.keras.backend.clear_session()

    if autoencoder is not None:
        inputs = {'input_image': input_numpy}
        try:
            autoencoder.get_layer('frac')
        except ValueError:
            pass
        else:
            inputs.update({'frac': tf.constant(1., shape=(batch_size,))})
        try:
            autoencoder.get_layer('distances')
        except ValueError:
            pass
        else:
            inputs.update({'distances': np.zeros_like(input_numpy)})
        gnina_input, _ = autoencoder.predict_on_batch(inputs)
    else:
        gnina_input = input_numpy

    if labels_tensor is None:  # We don't know labels; just return predictions
        return model.predict_on_batch(gnina_input)

    batch.extract_label(0, labels_tensor)  # y_true
    if train:  # Return loss
        return model.train_on_batch(
            gnina_input, labels_tensor.tonumpy())
    else:  # Return labels, predictions
        return (labels_tensor.tonumpy(),
                model.predict_on_batch(gnina_input))


def _calculate_ligand_distances(rec_channels, input_tensor, point_dist):
    """Calculate the minimum distance from any ligand electron density.

    For each grid point in a gnina input, this function will find the minimum
    distance to any part of the ligand input with a non-zero density.

    Note: the optimised C++ version of this function is provided along with
    python bindings under cpp/src/calculate_distances, in a function named
    calculate_distances in the calculate_distances module which comes packaged
    with gnina_tensorflow. It is about 70 times faster than this function when
    used on a 16 x 24 x 24 x 24 input, takes the same arguments and returns
    the same array. This python version is included for testing purposes.

    Arguments:
        rec_channels: the number of receptor channels in the gnina input
        input_tensor: a 4D numpy array with dimensions (channels, x, y, z)
        point_dist: the input grid resolution (Angstroms)

    Returns:
        3D numpy array of dimension (x, y, z) containing the minimum distance
        of each point to some ligand density.
    """
    output_shape = input_tensor.shape
    lig_tensor = input_tensor[rec_channels:, :, :, :]
    lig_tensor = np.sum(lig_tensor, axis=0)
    lig_tensor[np.where(lig_tensor > 0)] = 1.0
    result = np.zeros(output_shape, dtype='float32')
    x, y, z = output_shape[1:]

    for i in range(x):
        for j in range(y):
            for k in range(z):
                coords = np.array([i, j, k])
                cube_size = 3
                min_dist = np.inf
                while cube_size <= 2 * x + 1:
                    radius = cube_size // 2 + 1
                    imin = max(0, i - radius)
                    imax = min(x, i + radius)
                    jmin = max(0, j - radius)
                    jmax = min(y, j + radius)
                    kmin = max(0, k - radius)
                    kmax = min(z, k + radius)
                    origin = np.array([imin, jmin, kmin])

                    mask_cube = lig_tensor[imin:imax, jmin:jmax, kmin:kmax]
                    rel_i, rel_j, rel_k = np.where(mask_cube > 0)
                    if not len(rel_i):
                        cube_size += 2
                        continue
                    relative_coords = np.vstack([rel_i, rel_j, rel_k])
                    for idx in range(len(rel_i)):
                        abs_coords = relative_coords[:, idx] + origin
                        dist = np.linalg.norm(abs_coords - coords, 2)
                        min_dist = min(dist, min_dist)
                    result[:, i, j, k] = min_dist
                    break
    return result * point_dist
