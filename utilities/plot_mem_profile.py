"""Plot the memory usage of one or more autoencoder jobs."""

import argparse
import os
import socket
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from jack.utilities import upload_to_imgur
except ImportError:
    upload_to_imgur = None


def obtain_dataframe(raw_fnames, omit_fields=()):
    """Generate and sanitise dataframe given autoencoder working directories."""
    res = pd.DataFrame()
    for raw_fname in raw_fnames:
        fname = Path(raw_fname).expanduser()
        for log in fname.rglob('**/memory_history.txt'):
            df = pd.read_csv(log, sep=' ')
            df['Job'] = '/'.join(str(fname).split('/')[-4:])
            res = pd.concat([res, df], copy=False)
    res['Usage'] /= 1024 ** 3
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
        hue='Memory type', row='Job',
        kind="line", legend='brief',
        height=5, aspect=1, facet_kws=dict(
            sharex=True, sharey=False, legend_out=True)
    )
    g.legend._visible = False
    plt.legend()
    for ax in g.axes.ravel():
        ax.set_ylim([-0.02, 1.1 * ax.get_ylim()[1]])
        ax.set_xlim(left=0)
        title = ax.get_title().replace('Job = ', '')
        title += '\n({0}@{1})'.format(os.getenv('USER'), socket.gethostname())
        ax.set_title(title)
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
