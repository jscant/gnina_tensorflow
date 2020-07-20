#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:04:42 2020

@author: scantleb
"""


import glob
import os
import gnina_embeddings_pb2
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
    for idx, filename in enumerate(glob.iglob(os.path.join(directory, '*.bin'))):
        target = extract_filename(filename)
        print(idx, total_records, target)
        encodings = gnina_embeddings_pb2.protein()
        #target_path = encodings.path
        encodings.ParseFromString(open(filename, 'rb').read())
        for ligand_struct in encodings.ligand:
            embeddings[target][ligand_struct.path] = np.array(ligand_struct.embedding)
        total_records += len(embeddings[target])
    return embeddings


def get_label(path):
    fname = extract_filename(path).lower()
    return int(fname.find('chembl') != -1)


def get_embeddings_arr(directory):
    embeddings = deque()
    labels = deque()
    for idx, filename in enumerate(glob.iglob(os.path.join(directory, '*.bin'))):
        target = extract_filename(filename)
        print(idx, len(embeddings), target)
        encodings = gnina_embeddings_pb2.protein()
        encodings.ParseFromString(open(filename, 'rb').read())
        for ligand_struct in encodings.ligand:
            ligand_path = ligand_struct.path
            embeddings.append(np.array(ligand_struct.embedding))
            labels.append(get_label(ligand_path))
    return embeddings, np.array(labels)



def main():
    base_path = 'encodings/encodings_translated_dude_relative'
    embeddings_arr, labels = get_embeddings_arr(base_path)
    print(embeddings_arr[0].shape)
#    embeddings_arr = [np.random.rand(100, )]*10
#    labels = np.ones((10, ), dtype='int32')
    classifier = RFC(n_estimators=500)
    classifier.fit(embeddings_arr, labels)
    
    del embeddings_arr

if __name__ == '__main__':
    main()
    
