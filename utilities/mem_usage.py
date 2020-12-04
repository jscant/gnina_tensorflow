"""Monitors and records memory use of a process."""

import argparse
import subprocess
from pathlib import Path
from time import sleep


def get_mem_usage(pid):
    output = subprocess.run(
        ['pmap', pid],
        capture_output=True, check=True, shell=False).stdout
    return int(output.decode()[:-1].split('\n')[-1].split()[1][:-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pid', type=str, help='Process ID', nargs='*')
    parser.add_argument(
        '--output_fname', '-o', type=str, default='~/Desktop/mem.txt')
    args = parser.parse_args()

    if isinstance(args.pid, str):
        pids = [args.pid]
    else:
        pids = args.pid
    output_fname = Path(args.output_fname).expanduser().resolve()
    with open(output_fname, 'w') as f:
        f.write(' '.join(pids) + '\n')
    while True:
        s = ''
        for pid in pids:
            try:
                mem_usage = str(get_mem_usage(pid))
            except subprocess.CalledProcessError:
                print('Could not find process {}. Aborting.'.format(pid))
                exit(0)
            s += mem_usage + ' '
        with open(output_fname, 'a') as f:
            f.write(s[:-1] + '\n')
        sleep(1)
