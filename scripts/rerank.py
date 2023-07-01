import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "third_party"))


import torch
import logging

import argparse
from src.models import *
from src.datasets.test_dataset import TestDataset
import os
import numpy as np
from src.utils.utils import (write_to_hdf5, evaluate_ranked_predictions, write_recalls_to_txt,
                            extract_reranked_prediction, read_baseprediction_from_hdf5,
                            MockExtractor)



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--extractor", type=str, default="MockExtractor()")

parser.add_argument("--scorer", type=str, default=None)

parser.add_argument("--num_candidates", type=int, default=None,
                        help="number of top candidates to save for each query.")

parser.add_argument("--output_folder", type=str, default=None,
                        help="Folder where output is going to be saved.")

parser.add_argument("--output_base_name", type=str, default=None,
                        help="Basename of every output is going to be saved.")


parser.add_argument("--base_predictions_path", type=str, default=None,
                        help="path of the folder with database and query images.")

parser.add_argument("--dataset_path", type=str, default=None,
                        help="path of the folder with database and query images.")
parser.add_argument("--database_folder", type=str, default=None,
                        help="name of the folder with the reference images.")
parser.add_argument("--queries_folder", type=str, default=None,
                        help="name of the folder with queries images.")

parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")

parser.add_argument("--recall_values", type=str, default="[1, 5, 10, 100]",
                        help="list of integer to be evaluate with eval.")


args = parser.parse_args()

args.recall_values = eval(args.recall_values)
extractor = eval(args.extractor).to(args.device).eval()
scorer = eval(args.scorer).to(args.device).eval()


output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file = os.path.join(output_folder, f"{args.output_base_name}_reranking.h5")


dataset = TestDataset(args.dataset_path, 
                      args.database_folder, 
                      args.queries_folder, 
                      preprocessing=extractor.get_preprocessing() if hasattr(extractor, "get_preprocessing") else scorer.get_preprocessing())


base_predictions = read_baseprediction_from_hdf5(args.base_predictions_path)

rerankend_predictions, reranked_scores = extract_reranked_prediction(extractor, 
                                                                     scorer, 
                                                                     dataset, 
                                                                     base_predictions, 
                                                                     args.num_candidates, 
                                                                     args.device)

predictions_filenames = [[os.path.basename(dataset.database_paths[j]) for j in rerankend_predictions[i]] for i in range(dataset.queries_num)]
queries_filenames = [os.path.basename(dataset.queries_paths[i]) for i in range(dataset.queries_num)]


output = {'predictions': rerankend_predictions,
          'scores': reranked_scores,
          'reranked_candidates_filenames': predictions_filenames,
          'queries_filenames': queries_filenames}

write_to_hdf5(output_file, output)

recalls, recalls_str = evaluate_ranked_predictions(rerankend_predictions,
                                                   dataset.positives_per_query,
                                                   args.recall_values)


output_recalls_filepath = os.path.join(args.output_folder, f"{args.output_base_name}_recalls.txt")

    
write_recalls_to_txt(output_recalls_filepath, args.output_base_name, recalls)

logging.info(args.output_base_name)
logging.info(recalls_str)
logging.info(f"{recalls}")
