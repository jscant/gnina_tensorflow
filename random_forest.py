#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:04:42 2020

@author: scantleb
"""


import argparse
import glob
import os
import gnina_embeddings_pb2
import joblib
import numpy as np

from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier as RFC
from gnina_functions import Timer

def extract_filename(path, include_extension=False):
    filename = path.split('/')[-1]
    return filename if include_extension else filename.split('.')[0]


def get_embeddings(directory):
    embeddings = defaultdict(dict)
    total_records = 0
    for idx, filename in enumerate(glob.iglob(
            os.path.join(directory, '*.bin'))):
        target = extract_filename(filename)
        print(idx, total_records, target)
        encodings = gnina_embeddings_pb2.protein()
        #target_path = encodings.path
        encodings.ParseFromString(open(filename, 'rb').read())
        for ligand_struct in encodings.ligand:
            embeddings[target][ligand_struct.path] = np.array(
                ligand_struct.embedding)
        total_records += len(embeddings[target])
    return embeddings


def get_label(path):
    fname = extract_filename(path).lower()
    return int(fname.find('chembl') != -1)


def get_embeddings_arr(directory):
    embeddings = deque()
    labels = deque()
    for idx, filename in enumerate(glob.iglob(
            os.path.join(directory, '*.bin'))):
        target = extract_filename(filename)
        with open('log.rf', 'a') as f:
            f.write('{0} {1} {2}\n'.format(idx, len(embeddings), target))
        encodings = gnina_embeddings_pb2.protein()
        encodings.ParseFromString(open(filename, 'rb').read())
        for ligand_struct in encodings.ligand:
            ligand_path = ligand_struct.path
            embeddings.append(np.array(ligand_struct.embedding))
            labels.append(get_label(ligand_path))
    return np.array(embeddings), np.array(labels)



def main(args):
    base_path = args.base_path
    embeddings_arr, labels = get_embeddings_arr(base_path)
    classifier = RFC(n_estimators=500, oob_score=True, n_jobs=-1)
    classifier.fit(embeddings_arr, labels)
    with open('log.rf', 'a') as f:
        f.write('Training complete\n')
    # classifier = joblib.load('classifier.joblib.pkl') 
    # joblib.dump(classifier, 'classifier.joblib.pkl', compress=0)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str, help=
                        'Directory containing binary serialised encodings')
    args = parser.parse_args()
    main(args)
    
