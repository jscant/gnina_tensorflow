"""Plot the results of mem_usage.py"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from jack.utilities import upload_to_imgur
except ImportError:
    upload_to_imgur = None
from matplotlib import pyplot as plt
from utilities.gnina_functions import condense


def extract_working_dir(pid):
    """Retrieve the working directory of a process (pid) if present in logs."""
    pid = str(pid)
    gnina_tf_root = Path(__file__).expanduser().resolve().parents[1]
    if not gnina_tf_root.is_file():
        return
    with open(gnina_tf_root / 'process_ids.log', 'a') as f:
        for line in f.readlines():
            chunks = line.split()
            if len(chunks) == 2 and chunks[0] == pid:
                return Path(chunks[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', type=str, default='~/Desktop/mem.txt')
    parser.add_argument('--upload', '-u', action='store_true')
    args = parser.parse_args()

    fname = Path(args.fname).expanduser().resolve()

    plt.figure(figsize=(15, 7))
    plt.ticklabel_format(useOffset=False, style='plain')

    df = pd.read_csv(fname, sep=' ')
    df.fillna(-1, inplace=True)

    results = {}
    for pid, values in df.iteritems():
        vals = values.to_numpy()
        vals = vals[np.where(vals >= 0)][10:-5]
        x, y = condense(vals, gap=5)
        y /= 1e6
        results[pid] = (x, y, 3600 * (y[-1] - y[0]) / len(values))
        working_dir = extract_working_dir(pid)
        if working_dir is None:
            label = pid
        else:
            label = '/'.join(str(
                working_dir).split('/')[-4:-1]) + ' (PID={})'.format(pid)
        plt.plot(x, y, label=label)
        print('Increases per hour ({1}): {0:.4f} GB'.format(
            3600 * (y[-1] - y[0]) / len(values), pid))

    n_fields = len(results)
    if n_fields == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        axes = tuple([ax])
    else:
        cols = (n_fields + 1) // 2
        width = cols * 10
        fig, axes = plt.subplots(2, cols, figsize=(width, 10))

    for idx, (pid, info) in enumerate(results.items()):
        col = idx // 2
        row = idx % 2
        if n_fields > 2:
            ax = axes[row, col]
        else:
            ax = axes[row]
        ax.plot(info[0], info[1], label='Process ID: {}'.format(pid))
        ax.set_title('Increases per hour: {0:.4f} GB'.format(
            info[2]))
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory usage (GB)')
        ax.legend()
        ax.grid()
    plt.savefig('mem.png')

    if args.upload and upload_to_imgur is not None:
        print(upload_to_imgur('mem.png'))
