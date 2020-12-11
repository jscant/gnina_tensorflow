"""Plot the memory usage of one or more autoencoder jobs."""

import argparse
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from jack.utilities import upload_to_imgur
except ImportError:
    upload_to_imgur = None

USER = os.getenv('USER')
HOSTNAME = socket.gethostname()


def obtain_dataframe(raw_fnames, omit_fields=()):
    """Generate and sanitise dataframe given autoencoder working directories."""
    res = pd.DataFrame()
    for raw_fname in raw_fnames:
        fname = Path(raw_fname).expanduser()
        for log in fname.rglob('**/memory_history.txt'):
            df = pd.read_csv(log, sep=' ')
            df['Job'] = str(log.parent).replace(
                str(Path('~').expanduser()), '~')
            res = pd.concat([res, df], copy=False)
    res['Usage'] /= 1024 ** 3
    res = res[res['Time'] > 100]
    res['Time'] -= 100
    for field in omit_fields:
        res = res[res['Memory_type'] != field]
    res.rename(columns={'Usage': 'Usage (GB)', 'Memory_type': 'Memory type',
                        'Time': 'Time (s)'}, inplace=True)
    return res


def plot(df):
    """Plot a facetgrid of relplots with memory usage contained in dataframe."""
    g = sns.relplot(
        data=df,
        x='Time (s)', y='Usage (GB)',
        hue='Memory type', col='Job',
        kind="line", legend='brief',
        col_wrap=5,
        height=5, aspect=1, facet_kws=dict(
            sharex=False, sharey=True, legend_out=True)
    )
    g.legend._visible = False
    for ax in g.axes.ravel():
        ax.set_ylim([-0.02, 1.1 * ax.get_ylim()[1]])
        ax.set_xlim(left=0)
        title = ax.get_title().replace('Job = ', '')
        highest_y = -np.inf
        dy = 0
        dt = -1
        for line in ax.get_lines():
            x, y = line._x, line._y
            if len(y) < 2:
                continue
            max_y, min_y = np.amax(y), np.amin(y)
            if max_y > highest_y:
                highest_y = max_y
                dy = y[-1] - y[0]
                dt = np.amax(x) - np.amin(x)
        dydt = 3600 * (dy / dt)
        title += '\nChange per hour: {:.3f} GB'.format(dydt)
        ax.set_title(title)
        ax.legend()
    # g.fig.suptitle('({0}@{1})'.format(USER, HOSTNAME), fontsize=15)
    # plt.legend()
    plt.tight_layout()
    return g


def main(args):
    filenames = [args.filename] if isinstance(args.filename, str) else \
        args.filename
    omit_fields = [] if args.omit_fields is None else args.omit_fields
    df = obtain_dataframe(filenames, omit_fields=omit_fields)

    sns.set_style('whitegrid')
    g = plot(df)
    g.savefig('mem.png')

    if args.upload and upload_to_imgur is not None:
        print(upload_to_imgur('mem.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='*', type=str)
    parser.add_argument('--upload', '-u', action='store_true')
    parser.add_argument('--omit_fields', '-o', nargs='*', type=str)
    args = parser.parse_args()
    main(args)
