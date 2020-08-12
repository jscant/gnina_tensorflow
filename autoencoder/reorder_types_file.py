#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:05:17 2020

@author: scantleb
@brief: Reorders types files such that they are grouped by receptor, and are
thus made suitable as inputs to calculate_encodings.py
"""

import argparse


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
    with open(types_fname, 'r') as f:
        for line in f.readlines():
            chunks = line.split()
            lines.append(tuple(chunks))
    
    if order_by_ligand:
        criteria = lambda x: (x[1], x[2])
    else:
        criteria = lambda x: x[1]
        
    lines = sorted(lines, key=criteria)
    return '\n'.join([' '.join(line) for line in lines])
    

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
    args = parser.parse_args()
    
    if not args.in_place and args.output is None:
        raise RuntimeError(
            'Either an output filename must be specified or the --in_place flag must be used')
    if args.in_place and args.output is not None:
        raise RuntimeError(
            'Cannot specify output filename while using the --in_place flag')
    
    sorted_types = reorder(
        'data/small_chembl_test.types', order_by_ligand=args.by_ligand)
    
    if args.in_place:
        output = args.input
    else:
        output = args.output
    with open(output, 'w') as f:
        f.write(sorted_types)
    