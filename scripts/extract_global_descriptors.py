import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "third_party"))


import torch
import logging
import numpy as np

import argparse
from src.models import *
from src.datasets.test_dataset import TestDataset
import os
from src.utils.utils import (knn_ranking, write_to_hdf5, 
                         evaluate_ranked_predictions, write_recalls_to_txt,
                         extract_database_descriptors, extract_queries_descriptors, read_from_hdf5)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--extractor", type=str, default=None)

parser.add_argument("--output_base_name", type=str, default=None,
                        help="Basename of every output is going to be saved.")

parser.add_argument("--output_folder", type=str, default=None,
                        help="Basename of every output is going to be saved.")

parser.add_argument("--dataset_path", type=str, default=None,
                        help="path of the folder with database and query images.")

parser.add_argument("--database_folder", type=str, default=None,
                        help="name of the folder with the reference images.")

parser.add_argument("--queries_folder", type=str, default=None,
                        help="name of the folder with queries images.")

parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")

parser.add_argument("--batch_size", type=int, default=32, help="_")

parser.add_argument("--num_workers", type=int, default=4, help="_")

parser.add_argument("--n_chunks", type=int, default=4)

parser.add_argument("--chunks", type=str, default=None)

parser.add_argument("--perform_evaluation", type=int, default=1)

parser.add_argument("--recall_values", type=str, default="[1, 5, 10, 100]",
                        help="list of integer to be evaluate with eval.")

parser.add_argument("--num_candidates", type=int, default=100,
                        help="number of top candidates to save for each query.")

args = parser.parse_args()
args.recall_values = eval(args.recall_values)
args.perform_evaluation = (args.perform_evaluation != 0)
if args.chunks is None:
    args.chunks = [i for i in range(args.n_chunks)]
else:
    args.chunks = eval(args.chunks)
    if not isinstance(args.chunks, list):
        args.chunks = [args.chunks]


if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

extractor = eval(args.extractor).to(args.device).eval()

dataset_args=(args.dataset_path, args.database_folder, args.queries_folder, None, True)

dataset = TestDataset(dataset_args[0], dataset_args[1], dataset_args[2], 
                      preprocessing=extractor.get_preprocessing(), db_paths_func=dataset_args[3],
                      compute_positives=dataset_args[4])


n_imag_per_chunk = dataset.database_num // args.n_chunks

start_id = [i*n_imag_per_chunk for i in range(args.n_chunks)]

end_id = [i*n_imag_per_chunk  for i in range(1, args.n_chunks)] + [dataset.database_num]

chunks_filepaths = [os.path.join(args.output_folder, f"{args.output_base_name}_{i}_of_{args.n_chunks}.h5") for i in range(args.n_chunks)]


for i in range(args.n_chunks):
    if not i in args.chunks:
        continue

    indices = np.arange(start_id[i], end_id[i])
    output_file = chunks_filepaths[i]

    subdataset = torch.utils.data.Subset(dataset, indices)
    subdataset.database_num = len(indices)

    database_descriptors = extract_database_descriptors(extractor, 
                                                    subdataset, 
                                                    args.batch_size, 
                                                    args.num_workers, 
                                                    args.device)

    

    output = {'global_descriptors': database_descriptors,
          'index': indices}

    write_to_hdf5(output_file, output)


if args.perform_evaluation:
    queries_descriptors = extract_queries_descriptors(extractor, 
                                                      dataset, 
                                                      args.num_workers, 
                                                      args.device)
    
    predictions = []
    distances = []
    for chunk_file, s_id in zip(chunks_filepaths, start_id):
        database_descriptors = read_from_hdf5(chunk_file, "global_descriptors")

        dist, pred = knn_ranking(queries_descriptors, database_descriptors, args.num_candidates)

        pred += s_id

        predictions.append(pred)
        distances.append(dist)

    predictions = np.concatenate(predictions, 1)
    distances = np.concatenate(distances, 1)

    for i in range(len(predictions)):
        sort = np.argsort(distances[i])
        predictions[i, :] = predictions[i, sort]
        distances[i, :] = distances[i, sort]

    predictions = predictions[:, 0:args.num_candidates]
    distances = distances[:, 0:args.num_candidates]
    predictions_filenames = [[os.path.basename(dataset.database_paths[j]) for j in predictions[i]] for i in range(dataset.queries_num)]
    queries_filenames = [os.path.basename(dataset.queries_paths[i]) for i in range(dataset.queries_num)]

    output = {'top_candidates_index': predictions,
              'top_candidates_distances': distances,
              'top_candidates_filenames': predictions_filenames,
              'queries_filenames': queries_filenames}
    
    output_file = os.path.join(args.output_folder, f"{args.output_base_name}.h5")

    write_to_hdf5(output_file, output)

    recalls, recalls_str = evaluate_ranked_predictions(predictions,
                                                       dataset.positives_per_query,
                                                       args.recall_values)


    output_recalls_filepath = os.path.join(args.output_folder, f"{args.output_base_name}_recalls.txt")

    
    write_recalls_to_txt(output_recalls_filepath, args.output_base_name, recalls)

    logging.info(args.output_base_name)
    logging.info(recalls_str)
    logging.info(f"{recalls}")
else:
    logging.info(f"Completed extraction for chunks {args.chunks}.")
