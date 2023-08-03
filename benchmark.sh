export VPR_DS_PATH=/data/vg_datasets/retrieval/tokyo247/images/test/
export VPR_DB_FOLDER=database
export VPR_Q_FOLDER=queries_night
export VPR_BASE_NAME=tokyo247_queries_night
export VPR_BASE_PRED="outputs/cosplace_50_tokyo247_queries_night.h5"




python scripts/extract_global_descriptors.py --output_base_name=cosplace_50_$VPR_BASE_NAME --extractor="CosPlace()" --n_chunks=4 --chunks=0 --perform_evaluation=0 --batch_size=64 --num_workers=8 --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/extract_global_descriptors.py --output_base_name=cosplace_50_$VPR_BASE_NAME --extractor="CosPlace()" --n_chunks=4 --chunks=1 --perform_evaluation=0 --batch_size=64 --num_workers=8 --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/extract_global_descriptors.py --output_base_name=cosplace_50_$VPR_BASE_NAME --extractor="CosPlace()" --n_chunks=4 --chunks=2 --perform_evaluation=0 --batch_size=64 --num_workers=8 --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/extract_global_descriptors.py --output_base_name=cosplace_50_$VPR_BASE_NAME --extractor="CosPlace()" --n_chunks=4 --chunks=3 --perform_evaluation=0 --batch_size=64 --num_workers=8 --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/extract_global_descriptors.py --output_base_name=cosplace_50_$VPR_BASE_NAME --extractor="CosPlace()" --n_chunks=4 --chunks=-1 --perform_evaluation=1 --batch_size=64 --num_workers=8 --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER





python scripts/rerank.py --output_base_name=r2d2_$VPR_BASE_NAME --extractor="R2D2()" --scorer="RANSAC(ransacReprojThreshold=24)" --base_predictions_path=$VPR_BASE_PRED --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/rerank.py --output_base_name=d2net_$VPR_BASE_NAME --extractor="D2Net()" --scorer="RANSAC(ransacReprojThreshold=24)" --base_predictions_path=$VPR_BASE_PRED --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/rerank.py --output_base_name=superglue_$VPR_BASE_NAME --extractor="SuperPoint()" --scorer="SuperGlue()" --base_predictions_path=$VPR_BASE_PRED --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/rerank.py --output_base_name=transvpr_$VPR_BASE_NAME --extractor="TransVPR()" --scorer="RANSAC(ransacReprojThreshold=24)" --base_predictions_path=$VPR_BASE_PRED --output_folder="outputs" --dataset_path="/data/vg_datasets/retrieval/tokyo247/images/test/" --database_folder="database" --queries_folder="queries_night"

python scripts/rerank.py --output_base_name=patchnetvlad_$VPR_BASE_NAME --extractor="PatchNetVLAD()" --scorer='MultiScaleRANSAC(**{"ransac_configs": {"ransacReprojThreshold": 30.,  "ransacIter": 2000}, "num_levels": 3, "weights": [0.45, 0.15, 0.4]})' --base_predictions_path=$VPR_BASE_PRED --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/rerank.py --output_base_name=patchnetvlad_$VPR_BASE_NAME --extractor="PatchNetVLAD()" --scorer='RAPID()' --base_predictions_path=$VPR_BASE_PRED --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/rerank.py --output_base_name=cvnet_$VPR_BASE_NAME --scorer='CVNet()' --base_predictions_path=$VPR_BASE_PRED --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER

python scripts/rerank.py --output_base_name=loftr_$VPR_BASE_NAME --scorer='LoFTR()' --base_predictions_path=$VPR_BASE_PRED --output_folder="outputs" --dataset_path=$VPR_DS_PATH --database_folder=$VPR_DB_FOLDER --queries_folder=$VPR_Q_FOLDER