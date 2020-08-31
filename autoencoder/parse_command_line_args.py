import argparse
from pathlib import Path

import tensorflow as tf

from autoencoder.autoencoder_definitions import nonzero_mae, composite_mse, \
    zero_mae, nonzero_mse, zero_mse


class LoadConfigTrain(argparse.Action):
    """Class for loading argparse arguments from a config file."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Overloaded function; See parent class."""

        if values is None:
            return
        values = Path(values).expanduser()
        config = values.parents[1] / 'config'
        if not config.exists():
            print('No config file found in experiment''s base directory '
                  '({})'.format(config))
            print('Only specified command line args will be used.')
            namespace.load_model = values
            return
        args = ''
        with open(config, 'r') as f:
            for line in f.readlines():
                chunks = line.split()
                if not len(chunks):
                    continue
                if chunks[0] not in ['load_model',
                                     'absolute_save_path',
                                     'use_cpu',
                                     'binary_mask',
                                     'save_encodings',
                                     'verbose']:
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

        values = Path(values).expanduser()
        config = values.parents[1] / 'config'
        if not config.exists():
            print('No config file found in experiment''s base directory '
                  '({})'.format(config))
            print('Only specified command line args will be used.')
            namespace.load_model = values
            return
        args = ''
        namespace.binary_mask = False
        namespace.use_cpu = False
        with open(config, 'r') as f:
            for line in f.readlines():
                chunks = line.split()
                if not len(chunks):
                    continue
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
            'zero_mae': zero_mae
        }
    )

    # Bug with add_loss puts empty dict at the end of model._layers which
    # interferes with some functionality (such as
    # tf.keras.utils.plot_model)
    # noinspection PyProtectedMember
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
            'load_model', type=str, action=LoadConfigTrain, nargs='?',
            help='Load saved keras model. If specified, this should be the '
                 'directory containing the assets of a saved autoencoder. '
                 'If specified, the options are loaded from the config file '
                 'saved when the original model was trained; any options '
                 'specified in the command line will override the options '
                 'loaded from the config file.')
        parser.add_argument("--train", '-t', type=str, required=False)
        parser.add_argument('--encoding_size', '-e', type=int, required=False,
                            default=50)
        parser.add_argument('--iterations', '-i', type=int, required=False)
        parser.add_argument(
            '--save_interval', type=int, required=False, default=10000)
        parser.add_argument(
            '--model', '-m', type=str, required=False, default='single',
            help='Model architecture; one of single (SingleLayerAutoencoder' +
                 ') or dense (DenseAutoEncodcer)')
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
        parser.add_argument('--recmap', type=str, required=False)
        parser.add_argument('--ligmap', type=str, required=False)
    else:
        parser.add_argument(
            '--load_model', type=str, action=LoadConfigTest,
            help='Load saved keras model. If specified, this should be the '
                 'directory containing the assets of a saved autoencoder. '
                 'If specified, the options are loaded from the config file '
                 'saved when the original model was trained; any options '
                 'specified in the command line will override the options '
                 'loaded from the config file.')
        parser.add_argument("--test", '-t', type=str, required=False)

    parser.add_argument("--data_root", '-r', type=str, required=False,
                        default='')
    parser.add_argument(
        '--batch_size', '-b', type=int, required=False, default=16)
    parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.')
    parser.add_argument(
        '--use_cpu', '-g', action='store_true')
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument(
        '--verbose', action='store_true',
        help='Do not suppress deprecation messages and other tf warnings')
    args = parser.parse_args()

    autoencoder = None
    if args.load_model is not None:  # Load a model
        autoencoder = pickup(args.load_model)
    elif test_or_train == 'test':
        raise RuntimeError(
            'Please specify a model to use to calculate encodings.')

    return autoencoder, args
