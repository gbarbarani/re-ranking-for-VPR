
import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors


default_transforms = transforms.Compose([lambda path: Image.open(path).convert("RGB"),
                                         transforms.ToTensor(),
                                         lambda x: x[None]])


def open_image(path):
    return Image.open(path).convert("RGB")


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold=25, preprocessing=default_transforms, db_paths_func=None,
                 compute_positives=True):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.database_folder = os.path.join(dataset_folder, database_folder)
        self.queries_folder = os.path.join(dataset_folder, queries_folder)
        self.dataset_name = os.path.basename(dataset_folder)

        #debug
        import logging
        logging.info("debug")
        logging.info(f"{self.dataset_folder}")
        logging.info(f"{self.database_folder}")
        logging.info(f"{self.queries_folder}")
        logging.info(f"{self.dataset_name}")
        #
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        if not os.path.exists(self.database_folder):
            raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
        if not os.path.exists(self.queries_folder):
            raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")
        
        self.base_transform = preprocessing
        
        #### Read paths and UTM coordinates for all images.
        if db_paths_func is None:
            self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True),
                                         key=lambda p: os.path.basename(p))
        else:
            self.database_paths = db_paths_func()

        self.queries_paths  = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"),  recursive=True),
                                     key=lambda p: os.path.basename(p))
        

        if compute_positives:
            # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
            self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(np.float32)
            self.queries_utms  = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(np.float32)
        
            # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)

            #debug
            import logging
            logging.info("debug")
            logging.info(f"{self.database_utms.shape}")
            logging.info(f"{self.database_utms[:3]}")
            #

            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.positives_per_query = knn.radius_neighbors(self.queries_utms, 
                                                             radius=positive_dist_threshold,
                                                             return_distance=False)
        else:
            self.positives_per_query = None
        
        self.images_paths  = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]
        
        self.database_num = len(self.database_paths)
        self.queries_num  = len(self.queries_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        normalized_img = self.base_transform(image_path)
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return  (f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >")
    
    def get_positives(self):
        return self.positives_per_query

