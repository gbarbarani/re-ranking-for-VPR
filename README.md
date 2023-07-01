# Re-Ranking for VPR

Code for "Are Local Features All You Need for Cross-Domain Visual Place Recognition?" CVPR IMW 2023. 
The code will be released before the 18th of June 2023 (the conference date).

[[ArXiv](https://arxiv.org/abs/2304.05887)] [[BibTex](https://github.com/gbarbarani/re-ranking-for-VPR#cite)]


# Download Datasets

You can download and format the datasets following the instrunctions [here](https://github.com/gmberton/VPR-datasets-downloader) . Notice that many datasets are released only under request, SX-XL test database and queries (included our sf-xl night and occlusion) can be requested at this [link](https://github.com/gmberton/CosPlace).

The final dataset folder structure and filename format should be:

```
datasets --- database --- [subfolders] ---  @UTM_east@UTM_north...jpg
                                            @UTM_east@UTM_north...jpg
                                            ...
                                           
             queries  --- [subfolders] ---  @UTM_east@UTM_north...jpg
                                            @UTM_east@UTM_north...jpg
                                            ...
             ...
```

# Setup

The required Python packages are specified in **requirements.txt** . To install all the third part code run the following commands:


```
mkdir third_party
cd third_party

git clone https://github.com/naver/r2d2.git
git clone https://github.com/mihaidusmanu/d2-net.git
git clone https://github.com/magicleap/SuperGluePretrainedNetwork
git clone https://github.com/sungonce/CVNet.git
git clone https://github.com/QVPR/Patch-NetVLAD
git clone https://github.com/RuotongWANG/TransVPR-model-implementation

```

Download CVNet pretrained weights [here](https://drive.google.com/drive/folders/1VE2uG0bynk6-XfokjE13VFNVQRb-rQM8) and to save them inside **third_party/CVNet**. To download D2-Net pretrained weights follow the instructions [here](https://github.com/mihaidusmanu/d2-net#downloading-the-models). Other models already download the weights or they will perform the dowload the first time they run.


# Global Retrieval

To extract global embeddings and to perform global retrieval you can use **scripts\extract_global_descriptors.py**. The scripts takes the following main parameters:

* **output_base_name**: an identifier that will be added as prefix at every outputs file.
* **output_folder**: where to save the outputs.
* **extractor**: the global extractor class and parameters to be instantiated. The only option supported at the moment is CosPlace.
* **n_chunks**: default 1, can be usefull to increase it if the database embeddings do not fit the RAM.
* **dataset_path**: path to the folder that contains database and queries subfolders.
* **database_folder**: the database subfolder.
* **queries_folder**: the queries subfolder.

An example command usage is:

```
python scripts/extract_global_descriptors.py --output_base_name="cosplace_50_sf_xl_queries_v1" --extractor="CosPlace()" --n_chunks=4 --output_folder="outputs" --dataset_path="/data/sf_xl/processed/test/" --database_folder="database" --queries_folder="queries_v1"
```

You can find the outputs in the **output_folder**:

* **<output_base_name>.h5**: cotanins top_candidates_index [N_queries, N_candidates] dataset indeces of queries and predictions,  top_candidates_distances [N_queries, N_candidates] retrieval score, top_candidates_filenames [N_queries, N_candidates] filenames of top predictions, queries_filenames [N_queries] filenames of queries.
* **<output_base_name>_recalls.txt**: recalls of the global retrieval step.

The script offers also the possibility to extract in parallel and evaluate the top predictions subsequently with **--perform_evaluation** and **--chunks** parameters as it is shown in the example:

```
export CUDA_VISIBLE_DEVICES=0

python scripts/extract_global_descriptors.py --output_base_name="cosplace_50_sf_xl_queries_v1" --extractor="CosPlace()" --n_chunks=2 --chunks=1 --perform_evaluation=0 --output_folder="outputs" --dataset_path="/data/sf_xl/processed/test/" --database_folder="database" --queries_folder="queries_v1" >chunk0.out 2>chunk0.err &

export CUDA_VISIBLE_DEVICES=1

python scripts/extract_global_descriptors.py --output_base_name="cosplace_50_sf_xl_queries_v1" --extractor="CosPlace()" --n_chunks=2 --chunks=1 --perform_evaluation=0 --output_folder="outputs" --dataset_path="/data/sf_xl/processed/test/" --database_folder="database" --queries_folder="queries_v1" >chunk1.out 2>chunk1.err &

# After extraction is performed 

python scripts/extract_global_descriptors.py --output_base_name="cosplace_50_sf_xl_queries_v1" --extractor="CosPlace()" --n_chunks=2 --chunks=-1 --output_folder="outputs" --dataset_path="/data/sf_xl/processed/test/" --database_folder="database" --queries_folder="queries_v1"

```

# Re-ranking

To perform re-ranking run **scripts/rerank.py**. An example command is below, the file **benchmark.sh** provide all the command parameters to reproduce our experiments. It requires to specify which global retrivial top candidates presaved file to load with the parameter **base_predictions_path**.

```
python scripts/rerank.py --output_base_name="r2d2_sf_xl_queries_v1" --extractor="R2D2()" --scorer="RANSAC(ransacReprojThreshold=24)" --base_predictions_path="outputs/cosplace_50_sf_xl_queries_v1.h5" --output_folder="outputs" --dataset_path="/data/sf_xl/processed/test/" --database_folder="database" --queries_folder="queries_v1"
```

The outputs will be saved in the **output_folder**. 

**<output_base_name>_reranking.h5**:  predictions [N_queries, N_candidates] the re-ranked candidates, scores [N_queries, N_candidates] the re-ranking scores, reranked_candidates_filenames [N_queries, N_candidates] filenames of top predictions, queries_filenames [N_queries] filenames of queries.
**<output_base_name>_recalls.txt**: recalls of the re-ranking step.

# Results

* **Cospalce presaved top candidates and recalls**: coming soon...
* **Re-ranking presaved top candidates and recalls**: coming soon...

# Acknowledgements

This library make use of code taken from the following libraries:
* [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)

## Cite
Here is the bibtex to cite our paper
```
@InProceedings{Barbarani_2023_CVPR,
    author    = {Barbarani, Giovanni and Mostafa, Mohamad and Bayramov, Hajali and Trivigno, Gabriele and Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
    title     = {Are Local Features All You Need for Cross-Domain Visual Place Recognition?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
}
```
