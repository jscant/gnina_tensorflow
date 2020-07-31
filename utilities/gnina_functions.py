"""
Created on Fri Jul 10 14:47:44 2020

@author: scantleb
@brief: Utility functions for use in various machine learning models
"""

import time
import datetime

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


def extract_filename(path, include_extension=False):
    """Extracts filename from full path.
    
    Arguments:
        path: File path
        include_extension: When true, include text after last '.' in filename
    """
    filename = path.split('/')[-1]
    return filename if include_extension else filename.split('.')[0]


def beautify_config(config, fname=None):
    """Formats dictionary into two columns, sorted in alphabetical order.

    Arguments:
        config: any dictionary
        fname: string containing valid path to filename for writing beautified
            config to. If left blank, output will be printed to stdout.
    """
    output = 'Time of experiment generation: {:%d-%m-%Y %H:%M:%S}\n\n'.format(
        datetime.datetime.now())

    sorted_config = sorted(
        [(arg, value) for arg, value in config.items()], key=lambda x: x[0]
    )

    # Padding magic
    max_len = max([len(i) for i in [arg for arg, _ in sorted_config]]) + 4
    padding = ' ' * max_len

    for arg, value in sorted_config:
        output += '{0}:{1}{2}\n'.format(arg, padding[len(arg):], value)
    if fname is not None:
        with open(fname, 'w') as f:
            f.write(output[:-1])
    else:
        print(output)


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
                  labels_tensor=None, train=True):
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
        raise RuntimeError('Labels must be provided for backpropagation',
                           'if train == True')
    batch = example_provider.next_batch(input_tensor.shape[0])
    gmaker.forward(batch, input_tensor, 0, random_rotation=train)

    if labels_tensor is None:  # We don't know labels; just return predictions
        return model.predict_on_batch(input_tensor.tonumpy())

    batch.extract_label(0, labels_tensor)  # y_true
    if train:  # Return loss
        return model.train_on_batch(
            input_tensor.tonumpy(), labels_tensor.tonumpy())
    else:  # Return labels, predictions
        return (labels_tensor.tonumpy(),
                model.predict_on_batch(input_tensor.tonumpy()))