"""Monitors and records memory use of a process."""

import argparse
import subprocess
from pathlib import Path
from time import sleep

import psutil


def get_mem_usage(pid):
    output = subprocess.run(
        ['pmap', pid], capture_output=True, check=True, shell=False).stdout
    return int(output.decode()[:-1].split('\n')[-1].split()[1][:-1])


def get_processes(user=None):
    """Returns process ids of running python3 processes owned by user."""
    pids = []
    for pid in psutil.pids():
        p = psutil.Process(pid)
        if p.name() == 'python3' and \
                'mem_usage.py' not in ' '.join(p.cmdline()) and \
                p.username() == user and len(p.cmdline()) > 1 and \
                'gnina_tensorflow' in ' '.join(p.cmdline()):
            pids.append(str(pid))
    return pids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('user', type=str, help='Linux username')
    parser.add_argument(
        '--output_fname', '-o', type=str, default='~/Desktop/mem.txt')
    args = parser.parse_args()
    output_fname = Path(args.output_fname).expanduser().resolve()

    usages = ''
    pids = []
    while True:
        processes = get_processes(args.user)
        if not len(processes):
            break
        pids = sorted(list(set(processes + pids)))
        titles = ' '.join(pids) + '\n'
        s = ''
        for pid in pids:
            try:
                mem_usage = str(get_mem_usage(pid))
            except subprocess.CalledProcessError:
                pass
            else:
                s += mem_usage + ' '
        if len(pids):
            usages += s[:-1] + '\n'
            with open(output_fname, 'w') as f:
                f.write(titles + usages[:-1] + '\n')
        sleep(1)
