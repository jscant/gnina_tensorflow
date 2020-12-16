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
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from utilities.gnina_functions import Timer, format_time, write_process_info, \
    get_embeddings_arr


def main(args):
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

    save_path = Path(args.save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    # Logging process ID is useful for memory profiling (see utilities)
    write_process_info(__file__, save_path)

    if args.model_file is None and args.train_dir is not None:
        print('Extracting serialised embeddings from {}'.format(args.train_dir))
        with Timer() as t:
            embeddings_arr, labels, _ = get_embeddings_arr(args.train_dir)
        print('Embeddings sucessfully extracted ({}).\n'
              'Training classifier...'.format(format_time(t.interval)))

        classifier = RandomForestClassifier(
            n_estimators=500, oob_score=True, n_jobs=-1, verbose=2)
        with Timer() as t:
            classifier.fit(embeddings_arr, labels)
        print('Training complete ({}).'.format(format_time(t.interval)))

        model_filename = save_path / 'classifier.joblib'
        joblib.dump(classifier, model_filename, compress=0)
        print('Model saved as joblib object to {}'.format(model_filename))

        # Make memory available to garbage collection by reducing reference
        # count to zero (these arrays are large)
        del embeddings_arr, labels
    else:  # Load model file from disk
        with Timer() as t:
            classifier = joblib.load(str(Path(args.model_file).resolve()))
        print('Sucessfully loaded model from file: {0} ({1})'.format(
            args.model_file, format_time(t.interval)))

    if args.test_dir is None:  # Train only
        print('No test set provided; exiting.')
        return

    # Inference
    print('Extracting serialised embeddings from {}'.format(args.test_dir))
    with Timer() as t:
        test_embeddings, test_labels, paths = get_embeddings_arr(args.test_dir)
    print('Embeddings sucessfully extracted ({0}).\n'
          'Performing inference on {1}...'.format(format_time(t.interval),
                                                  args.test_dir))
    with Timer() as t:
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
    print('Inference finished in {}.'.format(format_time(t.interval)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_path', type=str,
                        help='Where to save trained model and predictions')
    parser.add_argument('--train_dir', '-t', type=str, required=False,
                        help='Directory containing binary serialised encodings '
                             'for training')
    parser.add_argument('--model_file', '-m', type=str, required=False,
                        help='Location of joblib object containing trained '
                             'model')
    parser.add_argument('--test_dir', '-v', type=str, required=False,
                        help='Directory containing binary serialised encodings '
                             'for testing')
    arguments = parser.parse_args()
    main(arguments)
