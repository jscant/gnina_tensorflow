#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:11:57 2020

@author: scantleb
@brief: Use trained autoencoder to calculate encodings for gnina inputs.
"""

import time
from pathlib import Path

import molgrid
import tensorflow as tf

from autoencoder.parse_command_line_args import parse_command_line_args
from utilities import gnina_embeddings_pb2, gnina_functions
from utilities.reorder_types_file import reorder


def calculate_encodings(encoder, data_root, batch_size, types_file, save_path,
                        dimension, resolution, rotate=False, ligmap=None,
                        recmap=None, binary_mask=False):
    """Calculates encodings for gnina inputs.

    Uses trained AutoEncoder object to calculate the encodings of all gnina
    inputs in types_file, which are then serialised to a protobuf message.

    Arguments:
        encoder: trained AutoEncoder object
        data_root: path to which all paths in types file are relative
        batch_size:
        types_file: contains list of gninatypes paths, storing the gnina-style
            inputs which are to be encoded in a lower dimensional space
        save_path:
        dimension:
        resolution:
        rotate: whether to randomly rotate gnina inputs in increments of 15
            degrees
        ligmap: Text file containing definitions of ligand input channels
        recmap: Text file containing definitions of receptor input channels
        binary_mask:

    Returns:
        Dictionary of serialised protein protobuf messages with structure
        defined in gnina_encodings.proto. The structure is:

            {receptor_path : serialised_protobuf_messages (1 per ligand)}
    """

    def get_paths(fname):
        """Reads types file to give path, label and indexing information.

        Arguments:
            fname: types file with lines of the format:
                <label> <receptor_path> <ligand_path>

        Returns a dictionary mapping of { global_idx: (label, rec, lig) } where
        global_idx is the position of the receptor/ligand pair in the types
        file, label is in {0, 1}, and rec and lig are the relative paths to
        the receptor and ligand gninatypes files, respectively.
        """
        result_paths = {}
        recs = set()
        curr_rec = None
        with open(fname, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                chunks = line.strip().split()
                lab = int(chunks[0])
                rec_path = chunks[1]
                lig_path = chunks[2]
                if idx == 0:
                    curr_rec = rec_path
                    recs.add(rec_path)
                elif rec_path != curr_rec:
                    if rec_path in recs:
                        # We have a types file unordered by receptor; create a
                        # temp ordered file, extract what we need, then delete.
                        reordered_types_file = reorder(fname)
                        tmp_types_file = save_path / 'tmp_types_file.types'
                        with open(tmp_types_file, 'w') as tmp:
                            tmp.write(reordered_types_file)
                        _, _, result_paths = get_paths(tmp_types_file)
                        return True, tmp_types_file, result_paths
                    else:
                        recs.add(rec_path)
                        curr_rec = rec_path
                result_paths[idx] = (lab, rec_path, lig_path)
        return False, fname, result_paths

    def write_encodings_to_disk(receptor, enc):
        """Write encodings to disk in serialised binary format.

        Arguments:
            receptor: path to receptor gninatypes file
            enc: iterable of tuples, each containing a label, a ligand path and
                a ligand encoding (numpy array)
        """
        rec_msg = gnina_embeddings_pb2.protein()
        rec_msg.path = receptor
        for lab, lig_path, lig_encoding in enc:
            lig_msg = rec_msg.ligand.add()
            lig_msg.path = lig_path
            lig_msg.embedding.extend(lig_encoding)
            lig_msg.label = lab

        fname = Path(receptor).stem + '.bin'
        with open(Path(save_path) / 'encodings' / fname, 'wb') as f:
            f.write(rec_msg.SerializeToString())

    types_file = Path(types_file).expanduser()
    save_path = Path(save_path).expanduser()
    encodings_dir = Path(save_path) / 'encodings'
    encodings_dir.mkdir(exist_ok=True, parents=True)

    delete_types_file, types_file, paths = get_paths(types_file)

    # Setup libmolgrid to feed Examples into tensorflow objects
    example_provider_kwargs = {
        'data_root': str(Path(data_root).expanduser()), 'balanced': False,
        'shuffle': False, 'cache_structs': False
    }
    if ligmap is None or recmap is None:
        # noinspection PyArgumentList
        e_test = molgrid.ExampleProvider(**example_provider_kwargs)
    else:
        recmap = Path(recmap).expanduser().resolve()
        ligmap = Path(ligmap).expanduser().resolve()
        rec_typer = molgrid.FileMappedGninaTyper(str(recmap))
        lig_typer = molgrid.FileMappedGninaTyper(str(ligmap))
        # noinspection PyArgumentList
        e_test = molgrid.ExampleProvider(
            rec_typer, lig_typer, **example_provider_kwargs)
    e_test.populate(str(Path(types_file).expanduser()))

    if delete_types_file:
        types_file.unlink()

    # noinspection PyArgumentList
    gmaker = molgrid.GridMaker(
        binary=binary_mask,
        dimension=dimension,
        resolution=resolution)

    # noinspection PyArgumentList
    dims = gmaker.grid_dimensions(e_test.num_types())
    tensor_shape = (batch_size,) + dims
    input_tensor = molgrid.MGrid5f(*tensor_shape)

    # Need a dictionary mapping {global_idx: (label, rec, lig) where global_idx
    # is the position of the receptor/ligand pair in the types file
    total_size = len(paths)
    iterations = total_size // batch_size

    # Inference (obtain encodings)
    current_rec = paths[0][1]
    encodings = []
    start_time = time.time()

    try:
        encoder.get_layer('frac')
    except ValueError:
        composite = False
    else:
        composite = True

    for iteration in range(iterations):

        # There is a memory leak with predict_on_batch(), this workaround seems
        # to prevent it.
        tf.keras.backend.clear_session()

        batch = e_test.next_batch(batch_size)
        gmaker.forward(batch, input_tensor, 0, random_rotation=rotate)

        inputs = [input_tensor.tonumpy()]
        if composite:  # We don't use this but is needed for a valid model
            inputs.append(tf.constant(1., shape=(batch_size,)))
        _, encodings_numpy = encoder.predict_on_batch(inputs)

        for batch_idx in range(batch_size):
            global_idx = iteration * batch_size + batch_idx
            label, rec, lig = paths[global_idx]
            if rec != current_rec:
                write_encodings_to_disk(current_rec, encodings)
                encodings = []
                current_rec = rec
            encodings.append((label, lig, encodings_numpy[batch_idx, :]))

        time_elapsed = time.time() - start_time
        time_per_iter = time_elapsed / (iteration + 1)
        time_remaining = time_per_iter * (iterations - iteration - 1)
        formatted_eta = gnina_functions.format_time(time_remaining)

        if not iteration:
            print('\n')

        console_output = ('Iteration: {0}/{1}\n' +
                          'Time elapsed {2} | Time remaining: {3}').format(
            iteration, iterations, gnina_functions.format_time(time_elapsed),
            formatted_eta)
        gnina_functions.print_with_overwrite(console_output)

    remainder = total_size % batch_size
    batch = e_test.next_batch(batch_size)
    gmaker.forward(batch, input_tensor, 0, random_rotation=rotate)

    inputs = [input_tensor.tonumpy()]
    if composite:
        inputs.append(tf.constant(1., shape=(batch_size,)))
    _, encodings_numpy = encoder.predict_on_batch(inputs)

    for batch_idx in range(remainder):
        global_idx = iterations * batch_size + batch_idx
        label, rec, lig = paths[global_idx]
        if rec != current_rec:
            write_encodings_to_disk(current_rec, encodings)
            encodings = []
            current_rec = rec
        encodings.append((label, lig, encodings_numpy[batch_idx, :]))

    if len(encodings):  # Encodings that have not been saved (final receptor)
        write_encodings_to_disk(current_rec, encodings)


if __name__ == '__main__':
    # Parse and sanitise command line args
    autoencoder, args = parse_command_line_args('test')
    autoencoder.summary()

    molgrid.set_gpu_enabled(1 - args.use_cpu)
    save_path = Path(args.save_path) / args.name

    tf.keras.backend.clear_session()

    with gnina_functions.Timer() as t:
        calculate_encodings(
            encoder=autoencoder, data_root=args.data_root,
            batch_size=args.batch_size, types_file=args.test,
            save_path=save_path, dimension=args.dimension,
            resolution=args.resolution, rotate=False, ligmap=args.ligmap,
            recmap=args.recmap, binary_mask=args.binary_mask)
    print('Encodings calculated and saved in {} s'.format(t.interval))
    print('Encodings written to {}'.format(save_path))
