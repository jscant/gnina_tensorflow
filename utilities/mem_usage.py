"""Monitors and records memory use of a process."""

import argparse
import subprocess
from pathlib import Path
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument('pid', type=str, help='Process ID')
parser.add_argument(
    '--output_fname', '-o', type=str, default='~/Desktop/mem.txt')
args = parser.parse_args()


def get_mem_usage(pid):
    output = subprocess.run(
        ['pmap', pid],
        capture_output=True, check=True, shell=False).stdout
    return int(output.decode()[:-1].split('\n')[-1].split()[1][:-1])


if __name__ == '__main__':
    output_fname = Path(args.output_fname).expanduser().resolve()
    with open(output_fname, 'w') as f:
        f.write('')
    while True:
        mem_usage = str(get_mem_usage(args.pid))
        with open(output_fname, 'a') as f:
            f.write(mem_usage + '\n')
        sleep(1)
