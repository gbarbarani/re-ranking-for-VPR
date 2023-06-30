import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing

import faiss
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import h5py



def extract_database_descriptors(extractor, dataset, batch_size=8, num_workers=4, device="cuda"):
    with torch.no_grad():
        database_subset_ds = Subset(dataset, list(range(dataset.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, 
                                     num_workers=num_workers,
                                     batch_size=batch_size, 
                                     pin_memory=(device=="cuda"),
                                     collate_fn=extractor.collate_fn,
                                     shuffle=False)

        database_descriptors = np.empty((len(database_subset_ds), extractor.output_dim), dtype="float32")

        for images, indices in tqdm(database_dataloader, ncols=100):
            indices = (indices % dataset.database_num)
            data = {"image": images.to(device)}
            out = extractor(data)
            descriptors = out["global_descriptors"]

            descriptors = descriptors.cpu().numpy()
            database_descriptors[indices.numpy(), :] = descriptors

        return database_descriptors
    
def extract_queries_descriptors(extractor, dataset, num_workers=4, device="cuda"):
    with torch.no_grad():
        queries_batch_size = 1
        queries_subset_ds = Subset(dataset, 
                                    list(range(dataset.database_num, 
                                            dataset.database_num+dataset.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, 
                                    num_workers=num_workers,
                                    batch_size=queries_batch_size, 
                                    pin_memory=(device=="cuda"),
                                    collate_fn=extractor.collate_fn,
                                    shuffle=False)
    
        queries_descriptors = np.empty((len(queries_subset_ds), extractor.output_dim), dtype="float32")
        for images, indices in tqdm(queries_dataloader, ncols=100):
            data = {"image": images.to(device)}

            out = extractor(data)
            descriptors = out["global_descriptors"]
            
            descriptors = descriptors.cpu().numpy()
        
            indices = indices.numpy() - dataset.database_num
            queries_descriptors[indices, :] = descriptors

    return queries_descriptors




def extract_global_descriptors(extractor, dataset, batch_size=8, num_workers=4, device="cuda"):
    with torch.no_grad():
        database_subset_ds = Subset(dataset, list(range(dataset.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, 
                                     num_workers=num_workers,
                                     batch_size=batch_size, 
                                     pin_memory=(device=="cuda"),
                                     collate_fn=extractor.collate_fn)

        all_descriptors = np.empty((len(dataset), extractor.output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            data = {"image": images.to(device)}
            
            out = extractor(data)
            descriptors = out["global_descriptors"]

            descriptors = descriptors.cpu().numpy()
        
            all_descriptors[indices.numpy(), :] = descriptors

        queries_batch_size = 1
        queries_subset_ds = Subset(dataset, 
                                   list(range(dataset.database_num, 
                                              dataset.database_num+dataset.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, 
                                    num_workers=num_workers,
                                    batch_size=queries_batch_size, 
                                    pin_memory=(device=="cuda"),
                                    collate_fn=extractor.collate_fn)
    
        for images, indices in tqdm(queries_dataloader, ncols=100):
            data = {"image": images.to(device)}

            out = extractor(data)
            descriptors = out["global_descriptors"]
            
            descriptors = descriptors.cpu().numpy()
        
            all_descriptors[indices.numpy(), :] = descriptors

        queries_descriptors = all_descriptors[dataset.database_num:]
        database_descriptors = all_descriptors[:dataset.database_num]

    return queries_descriptors, database_descriptors


def knn_ranking(queries_descriptors, database_descriptors, num_candidates=100):
    embedding_dim = database_descriptors.shape[1]

    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(database_descriptors)


    distances, predictions = faiss_index.search(queries_descriptors, num_candidates)

    return distances, predictions

def write_to_hdf5(output_file, output):
    h5f = h5py.File(output_file, 'w')
    for key, val in output.items():
        h5f.create_dataset(key, data=val)
    
    h5f.close()

def read_baseprediction_from_hdf5(input_file):
        h5f = h5py.File(input_file, 'r')
        base_predictions = h5f["top_candidates_index"][()]
        h5f.close()

        return base_predictions

def read_from_hdf5(input_file, keys):
        if not (isinstance(keys, list) or isinstance(keys, tuple)):
            keys = (keys,)

        h5f = h5py.File(input_file, 'r')
        output = [h5f[k][()] for k in keys]
        h5f.close()

        if len(output) == 1:
            output = output[0]

        return output

def evaluate_ranked_predictions(predictions, positives_per_query, recall_values=[1, 5, 10]):
    recalls = np.zeros(len(recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break

    recalls = recalls / len(positives_per_query) * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])

    return recalls, recalls_str

    
def write_recalls_to_txt(filepath, dataset_name, recalls):
    with open(filepath, 'a') as file:
        file.write(f"{dataset_name}: {recalls} \n")


def extract_reranked_prediction(extractor, scorer, dataset, base_predictions, num_candidates=100, device="cuda"):


    with torch.no_grad():
        database_subset_ds = Subset(dataset, list(range(dataset.database_num)))
        queries_subset_ds = Subset(dataset, list(range(dataset.database_num, dataset.database_num+dataset.queries_num)))

        rerankend_predictions = []
        reranked_scores = []
        for query_index in tqdm(range(len(base_predictions)), ncols=100):
            query_image, _ = queries_subset_ds[query_index]

            query_image = query_image.to(device)
            query_out = extractor({"image": query_image})

            preds_with_scores = []
            b_predictions = base_predictions[query_index][:num_candidates]
            for ref_index in b_predictions:
                ref_image, _ = database_subset_ds[ref_index]

                ref_image = ref_image.to(device)

                ref_out = extractor({"image": ref_image})
                data = {"image0": query_image, "image1":ref_image}
                data = {**data, **{k+"0" : v for k, v in query_out.items()}, **{k+"1": v for k, v in ref_out.items()}}

                scorer_output = scorer(data)

                score = scorer_output["score"]

                preds_with_scores.append((ref_index, score))

            preds_with_scores.sort(key=lambda a: a[1])
            r_predictions, r_scores = zip(*preds_with_scores)

            rerankend_predictions.append(r_predictions)
            reranked_scores.append(r_scores)

        return rerankend_predictions, reranked_scores


class MockExtractor():
    def __call__(self, x):
        return {}
    def to(self, device):
        return self
    def eval(self):
        return self


def download_from_drive(id, dest):
    import gdown
    
    url = "https://drive.google.com/u/0/uc?id=1yNzxsMg34KO04UJ49ncANdCIWlB3aUGA"
    output = "resnet50_2048.pth"
    gdown.download(url, output, quiet=False)