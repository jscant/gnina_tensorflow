#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:05:17 2020

@author: scantleb
@brief: Reorders types files such that they are either grouped by receptor, and
are thus made suitable as inputs to calculate_encodings.py, or shuffled into a
random order.
"""

import argparse

import numpy as np


def reorder(types_fname, order_by_ligand=False):
    """Reorders types file by the receptor, and then optionally the ligand.
    
    Sorting is lexicographic.
    
    Arguments:
        types_fname: File containing types information (label rec ligand)
        order_by_ligand: Use secondary sorting to sort by ligand after sorting
            by receptor.
    Returns:
        String containing lines sorted by the receptor (and optionally ligand)
        field.
    """
    lines = []
    with open(types_fname, 'r') as fname:
        for line in fname.readlines():
            chunks = line.split()
            lines.append(tuple(chunks))

    if order_by_ligand:
        criteria = lambda x: (x[1], x[2])
    else:
        criteria = lambda x: x[1]

    lines = sorted(lines, key=criteria)
    return '\n'.join([' '.join(line) for line in lines])


def shuffle(types_fname):
    """Shuffles a types file into a random order.

    Arguments:
        types_fname: File containing types information (label rec ligand)

    Returns:
        String containing lines of input file shuffled randomly.
    """
    with open(types_fname, 'r') as fname:
        lines = list(fname.readlines())

    np.random.shuffle(lines)
    return ''.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input types filename')
    parser.add_argument('output', type=str, nargs='?',
                        help='Output types filename')
    parser.add_argument('--in_place', action='store_true',
                        help='Modify types file in place.')
    parser.add_argument('--by_ligand', action='store_true',
                        help="""Order first by receptor, then by ligand.
                        If this is not specified, there is no order within
                        contiguous receptor information.""")
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle (rather than sort) output.')
    args = parser.parse_args()

    if not args.in_place and args.output is None:
        raise RuntimeError(
            'Either an output filename must be specified or the --in_place '
            'flag must be used')
    if args.in_place and args.output is not None:
        raise RuntimeError(
            'Cannot specify output filename while using the --in_place flag')
    if args.by_ligand and args.shuffle:
        raise RuntimeError(
            '--by_ligand and --shuffle are not compatible with one another.'
        )

    if args.shuffle:
        new_order = shuffle(args.input)
    else:
        new_order = reorder(args.input, order_by_ligand=args.by_ligand)

    if args.in_place:
        output = args.input
    else:
        output = args.output
    with open(output, 'w') as f:
        f.write(new_order)
