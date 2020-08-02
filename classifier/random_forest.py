"""
Created on Mon Jul 20 15:04:42 2020

@author: scantleb
@brief: Trains and tests random forests on gnina embeddings.

Gnina embeddings generated using the gnina framework
(https://github.com/gnina/gnina) and saved in protobuffer serialised format
are used to train and test a random forest. This is intended as a sanity check
for the expressiveness of the penultimate layer of a trained gnina model.
"""


import argparse
import gnina_embeddings_pb2
import joblib
import numpy as np
import os
from pathlib import Path

from collections import deque
from gnina_functions import Timer
from sklearn.ensemble import RandomForestClassifier


def get_embeddings_arr(directory):
    """Reads serialised embeddings in from protobuffer format.

    Directory should contain .bin files generated using gnina_tensoflow.py.
    Each serialised .bin file should contain one protein message, which in
    turn should contain all the ligand messages associated with that target.

    Arguments:
        directory: Directory containing serialised gnina embeddings, as
            defined in gnina_embeddings.proto.
    
    Returns:
        Numpy arrays containing both the embeddings and the labels, along with
        a dictionary mapping the index of each embedding/label along to a
        tuple containing the relative path of the protein and ligand gninatypes
        file used to generate that embedding.
    """
    embeddings = deque()
    labels = deque()
    paths = deque()
    
    for idx, filename in enumerate(Path(directory).glob('*.bin')):
        encodings = gnina_embeddings_pb2.protein()
        encodings.ParseFromString(open(filename, 'rb').read())
        target_path = encodings.path
        for ligand_struct in encodings.ligand:
            label = ligand_struct.label
            embeddings.append(np.array(ligand_struct.embedding))
            labels.append(label)
            ligand_path = ligand_struct.path
            paths.append((target_path, ligand_path))
    path_dict = { idx : path_tup for idx, path_tup in enumerate(paths) }
    return np.array(embeddings), np.array(labels), path_dict


def main(args):
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    if not isinstance(slurm_job_id, str):
        slurm_job_id = 'noslurm_{}'.format(np.random.randint(0, 1000000))
        
    # Sanitise arguments
    if args.train_dir is not None and args.model_file is not None:
        raise RuntimeError(
            'Only one of train_dir and model_file can be specified')
    if args.train_dir is None and args.model_file is None:
        raise RuntimeError(
            'Please specify one either train_dir or model_file')
    if args.train_dir is None and args.test_dir is None:
        raise RuntimeError(
            'Please specify either train_dir, test_dir, or both')
        
    save_path = Path(args.save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    if args.model_file is None and args.train_dir is not None:
        embeddings_arr, labels, _ = get_embeddings_arr(args.train_dir)
        print('Embeddings sucessfully extracted')
        
        classifier = RandomForestClassifier(
            n_estimators=500, oob_score=True, n_jobs=-1)
        with Timer() as t:
            classifier.fit(embeddings_arr, labels)
        print('Training complete in {0:0.3f} s'.format(t.interval))
    
        model_filename = save_path / 'classifier.joblib'
        joblib.dump(classifier, model_filename, compress=0)
        print('Model saved as joblib object to {}'.format(model_filename))
        
        # Make memory available to garbage collection by reducing reference
        # count to zero (these arrays are large)
        del embeddings_arr, labels
    elif args.model_file is not None: # Load model file from disk
        classifier = joblib.load(str(Path(args.model_file).resolve()))
        print('Sucessfully loaded model from file: {}'.format(args.model_file))
    
    if args.test_dir is None: # Train only
        print('No test set provided; exiting.')
        return
    
    # Inference
    test_embeddings, test_labels, paths = get_embeddings_arr(args.test_dir)
    test_predictions = classifier.predict_proba(test_embeddings)
    
    # Write inference results to disk in the standard gnina format
    predictions_filename = save_path / 'predictions_{}.txt'.format(
        Path(args.test_dir).stem)
    
    with open(predictions_filename, 'w') as f:
        f.write('')
    
    predictions_output = ''
    for idx in range(len(test_predictions)):
        pred = test_predictions[idx][1]
        true = test_labels[idx]
        rec, lig = paths[idx]
        predictions_output += '{0} | {1:0.7f} {2} {3}\n'.format(
            true, pred, rec, lig)
        if not (idx + 1) % 1000:
            with open(predictions_filename, 'a') as f:
                f.write(predictions_output)
            predictions_output = ''
    
    with open(predictions_filename, 'a') as f:
        f.write(predictions_output[:-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str, help=
                        'Where to save trained model and predictions')
    parser.add_argument('--train_dir', '-t', type=str, required=False,
        help='Directory containing binary serialised encodings for training')
    parser.add_argument('--model_file', '-m', type=str, required=False,
        help='Location of joblib object containing trained model')
    parser.add_argument('--test_dir', '-v', type=str, required=False,
        help='Directory containing binary serialised encodings for testing')
    args = parser.parse_args()
    main(args)
    
