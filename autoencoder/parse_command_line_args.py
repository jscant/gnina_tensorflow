import argparse
from pathlib import Path

import tensorflow as tf

from autoencoder import autoencoder_definitions


def str_to_type(arg):
    """Determine if a string can be converted into an int or float

    Arguments:
        arg: (string)

    Returns:
        str, float or int version of the input, depending on whether the input
        is numeric and then whether it contains a decimal point.
    """
    try:
        float(arg)
    except ValueError:
        if arg == 'None':
            return None
        return str(arg)
    if arg.find('.') == -1 and arg.find('e') == -1:
        return int(arg)
    return float(arg)


class LoadConfigTrain(argparse.Action):
    """Class for loading argparse arguments from a config file."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Overloaded function; see parent class."""

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
        with open(config, 'r') as f:
            for line in f.readlines():
                chunks = line.split()
                if not len(chunks) or chunks[0] == 'resume':
                    continue
                if len(chunks) == 1:
                    setattr(namespace, chunks[0], True)
                elif chunks[1] in ['True', 'False']:
                    setattr(
                        namespace, chunks[0],
                        [False, True][chunks[1] == 'True'])
                elif chunks[0] == 'name':
                    setattr(namespace, *chunks)
                elif chunks[0] not in ['load_model',
                                       'absolute_save_path']:
                    setattr(namespace, chunks[0], str_to_type(chunks[1]))
                elif chunks[1] == 'None':
                    setattr(namespace, chunks[0], None)

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

        bin_opts = ['use_cpu', 'binary_mask', 'nesterov']
        for bin_opt in bin_opts:
            setattr(namespace, bin_opt, False)
        with open(config, 'r') as f:
            for line in f.readlines():
                chunks = line.split()
                if not len(chunks):
                    continue
                if chunks[0] in ['data_root', 'save_path', 'ligmap', 'recmap']:
                    setattr(namespace, *chunks)
                elif chunks[0] in ['dimension', 'resolution']:
                    setattr(namespace, chunks[0], float(chunks[1]))
                elif chunks[0] in ['batch_size']:
                    setattr(namespace, chunks[0], int(chunks[1]))
                elif len(chunks) == 1 or chunks[1] == 'True':
                    # store_true args present a problem, loaded manually
                    if chunks[0] in bin_opts:
                        setattr(namespace, chunks[0], True)

        # args.load_model is always None if we do not do this, even when
        # it is specified using --load_model.
        setattr(namespace, 'load_model', values)


def pickup(path):
    """Loads saved autoencoder.

    Arguments:
        path: location of saved weights and architecture

    Returns:
        AutoEncoderBase-derived object initialised with weights from saved
        checkpoint.
    """
    custom_objects = {
        'zero_mse': autoencoder_definitions.zero_mse,
        'nonzero_mse': autoencoder_definitions.nonzero_mse,
        'composite_mse': autoencoder_definitions.composite_mse,
        'nonzero_mae': autoencoder_definitions.nonzero_mae,
        'zero_mae': autoencoder_definitions.zero_mae,
        'trimmed_nonzero_mae': autoencoder_definitions.trimmed_nonzero_mae,
        'trimmed_zero_mae': autoencoder_definitions.trimmed_zero_mae,
        'close_mae': autoencoder_definitions.close_mae,
        'close_nonzero_mae': autoencoder_definitions.close_nonzero_mae,
        'close_zero_mae': autoencoder_definitions.close_zero_mae,
        'proximity_mse': autoencoder_definitions.proximity_mse,
        'enc_loss_fn': autoencoder_definitions.enc_loss_fn,
        'disc_loss_fn': autoencoder_definitions.disc_loss_fn,
        'SquaredError': autoencoder_definitions.SquaredError,
        'define_discriminator': autoencoder_definitions.define_discriminator
    }
    ae = tf.keras.models.load_model(path, custom_objects=custom_objects)
    return ae


def parse_command_line_args(test_or_train='train'):
    """Parse command line args and return as dict.

    Returns a namespace containing all args, default or otherwise; if 'pickup'
    is specified, as many args as are contained in the config file for that
    (partially) trained model are loaded, otherwise defaults are given.
    Command line args override args found in the config of model found in
    'pickup' directory. Also returns either None (no loaded model) or a
    keras loaded model (if loaded model is specified).
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
        parser.add_argument('--resume', action='store_true',
                            help='Loaded model is resumed as if it had never '
                                 'been stopped. Losses are appended to the end '
                                 'of the existing loss_log.txt and the '
                                 'iteration count starts where the loaded '
                                 'model left off.')
        parser.add_argument("--train", '-t', type=str, required=False)
        parser.add_argument('--encoding_size', '-e', type=int, required=False,
                            default=50)
        parser.add_argument('--iterations', '-i', type=int, required=False)
        parser.add_argument('--batch_size', '-b', type=int, required=False,
                            default=16, help='Number of examples per batch')
        parser.add_argument(
            '--save_interval', type=int, required=False, default=-1)
        parser.add_argument(
            '--model', '-m', type=str, required=False, default='single',
            help='Model architecture; one of single (SingleLayerAutoencoder' +
                 '), multi (MultiLayerAutoEncoder), dense '
                 '(DenseAutoEncodcer) or res (ResidualAutoEncoder).')
        parser.add_argument(
            '--optimiser', '-o', type=str, required=False, default='sgd')
        parser.add_argument(
            '--learning_rate', '-l', type=float, required=False)
        parser.add_argument(
            '--learning_rate_schedule', type=str, required=False, default=None,
            help='Learning rate schedule to use, one of "1cycle", '
                 '"warm_restarts" or "stepwise". Only compatible with adamw '
                 'and sgdw optimisers.'
        )
        parser.add_argument(
            '--min_lr', type=float, required=False, default=-1.,
            help='Min learning rate for 1-cycle learning rate scheduling')
        parser.add_argument(
            '--max_lr', type=float, required=False, default=-1.,
            help='Max learning rate for 1-cycle learning rate scheduling')
        parser.add_argument(
            '--lrs_period', type=int, required=False, default=2000,
            help='Period over which to repeat the "warm_restarts" learning '
                 'rate pattern, or the number of iterations before the '
                 'learning rate drops each time for the "stepwise" schedule.'
        )
        parser.add_argument(
            '--lrs_beta', type=float, required=False, default=1.0,
            help='Beta value for "warm_restarts" learning rate schedule.'
        )
        parser.add_argument(
            '--momentum', type=float, required=False, default=0.0)
        parser.add_argument(
            '--nesterov', action='store_true',
            help='Use nesterov momentum (RMSProp and SGD(W) only)'
        )
        parser.add_argument(
            '--loss', type=str, required=False, default='mse')
        parser.add_argument(
            '--hidden_activation', type=str, required=False, default='sigmoid',
            help='Activation function for hidden layers')
        parser.add_argument(
            '--encoding_activation', type=str, required=False, default='linear',
            help='Activation function for encoding layers')
        parser.add_argument(
            '--conv_filters', type=int, required=False, default=1024,
            help='Filters per convolutional layer')
        parser.add_argument(
            '--final_activation', type=str, required=False, default='sigmoid')
        parser.add_argument('--binary_mask', action='store_true')
        parser.add_argument('--metric_distance_threshold', type=float,
                            required=False, default=-1.0,
                            help='If specified, a set of distance-thresholded '
                                 'metrics will be reported')
        parser.add_argument(
            '--dimension', type=float, required=False, default=18.0)
        parser.add_argument(
            '--resolution', type=float, required=False, default=1.0)
        parser.add_argument(
            '--save_encodings', action='store_true')
        parser.add_argument(
            '--overwrite_checkpoints', action='store_true',
            help='Each saved model state overwrites last'
        )
        parser.add_argument('--recmap', type=str, required=False)
        parser.add_argument('--ligmap', type=str, required=False)
        parser.add_argument(
            '--verbose', action='store_true',
            help='Do not suppress deprecation messages and other tf warnings')
        parser.add_argument('--adversarial', action='store_true',
                            help='Use adversarial CNN to make reconstructions '
                                 'more realistic.')
        parser.add_argument('--denoising', type=float, default=-1.0,
                            help='Rate of random zero-ing of inputs.')
        parser.add_argument('--adversarial_variance', type=float,
                            default=10.0)
    else:
        parser.add_argument(
            'load_model', type=str, action=LoadConfigTest, nargs='?',
            help='Load saved keras model. If specified, this should be the '
                 'directory containing the assets of a saved autoencoder. '
                 'If specified, the options are loaded from the config file '
                 'saved when the original model was trained; any options '
                 'specified in the command line will override the options '
                 'loaded from the config file.')
        parser.add_argument("--test", '-t', type=str, required=False)
        parser.add_argument('--statistics', action='store_true',
                            help='Generate and save statistics on '
                                 'reconstructions')

    parser.add_argument("--data_root", '-r', type=str, required=False,
                        default='')
    parser.add_argument(
        '--save_path', '-s', type=str, required=False, default='.')
    parser.add_argument(
        '--use_cpu', '-g', action='store_true')
    parser.add_argument('--name', type=str, required=False)
    args = parser.parse_args()

    autoencoder = None
    if args.load_model is not None:  # Load a model
        autoencoder = pickup(args.load_model)
    elif test_or_train == 'test':
        raise RuntimeError(
            'Please specify a model to use to calculate encodings.')

    return autoencoder, args
