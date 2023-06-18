import sys
from pathlib import Path
import torch
from torchvision import transforms
from .base_model import BaseModel

from CVNet.model.CVNet_Rerank_model import CVNet_Rerank


import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2


@torch.no_grad()
def extract_feature(model, im_list):
    with torch.no_grad():
        img_feats = [[] for i in range(len(im_list))] 

        for idx in range(len(im_list)):
            desc = model.extract_global_descriptor(im_list[idx])
            if len(desc.shape) == 1:
                desc.unsqueeze_(0)
            desc = F.normalize(desc, p=2, dim=1)
            img_feats[idx].append(desc.detach().cpu())

        for idx in range(len(img_feats)):
            img_feats[idx] = torch.cat(img_feats[idx], dim=0)
            if len(img_feats[idx].shape) == 1:
                img_feats[idx].unsqueeze_(0)

        img_feats_agg = F.normalize(torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), p=2, dim=1)
        img_feats_agg = img_feats_agg.cpu().numpy()

    return img_feats_agg 

class TensorDict(dict):
    def to(self, device):
        for k, v in self.items():
            self[k] = v.to(device)
        
        return self
    
class TensorList(list):
    def to(self, device):
        for i in range(len(self)):
            self[i] = self[i].to(device)
        
        return self
    

def color_norm(im, mean, std):
    """Performs per-channel normalization (CHW format)."""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

def prepare_im(im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = color_norm(im, _MEAN, _SD)
        return im


class Cv2ReadAndRescale():
    def __init__(self, scale_list) -> None:
        self.scale_list = scale_list

        self.tensor_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        lambda x: x[None]
                                        ])


    def __call__(self, path):
        im = cv2.imread(path)

        im_list = TensorList()

        for scale in self.scale_list:
            if scale == 1.0:
                im_np = im.astype(np.float32, copy=False)
            elif scale < 1.0:
                im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                im_np = im_resize.astype(np.float32, copy=False)
            elif scale > 1.0:
                im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                im_np = im_resize.astype(np.float32, copy=False)

            im_np = torch.FloatTensor(prepare_im(im_np))[None]
        
            im_list.append(im_np)  

        im = torch.FloatTensor(prepare_im(im.astype(np.float32, copy=False)))[None]

        return TensorDict(original=im, rescaled=im_list)



class CVNet(BaseModel):
    default_conf = {
        "MODEL_DEPTH": 50,
        "MODEL_HEADS_REDUCTION_DIM": 2048,
        "TEST_WEIGHTS": "models/cvnet/CVPR2022_CVNet_R50.pyth",
        "TEST_SCALE_LIST": [0.7071, 1.0, 1.4142],
        "mode": "reranking",
    }

    required_inputs = []

    def _init(self, conf):
    
        self.model = CVNet_Rerank(conf["MODEL_DEPTH"], conf["MODEL_HEADS_REDUCTION_DIM"]).eval()
        load_checkpoint(conf["TEST_WEIGHTS"], self.model)

        self.output_dim = conf["MODEL_HEADS_REDUCTION_DIM"]

    def get_preprocessing(self):
        return Cv2ReadAndRescale(self.conf["TEST_SCALE_LIST"])
    

    def collate_fn(self, batch):
        t_dict_list, indexes = zip(*batch)

        im_list = list(zip(*[td["rescaled"] for td in t_dict_list]))

        t_im_list = TensorList()
        for e in im_list:
            t_im_list.append(torch.cat(e))

        indexes = torch.LongTensor(indexes)

        return t_im_list, indexes


    def forward(self, data):
        
        
        if self.conf["mode"] == "reranking":
            out = {}

            image_q, im_list_q = data["image0"]["original"], data["image0"]["rescaled"]
            image_db, im_list_db = data["image1"]["original"], data["image1"]["rescaled"]
        

            Q = extract_feature(self.model, im_list_q)
            X = extract_feature(self.model, im_list_db)

            global_score = np.dot(X, Q.T)[0][0]
        

            feat_q = self.model.extract_featuremap(image_q)
            feat_db = self.model.extract_featuremap(image_db)


            reranking_score = self.model.extract_score_with_featuremap(feat_q, feat_db).cpu().numpy()

            score = global_score + 0.5 * reranking_score

            out["global_score"] = global_score
            out["reranking_score"] = reranking_score
        
            out["score"] = -score

            return out

        elif self.conf["mode"]=="global":
            out = {}
            im_list = data["image"]
        
            X = extract_feature(self.model, im_list)
            X = torch.FloatTensor(X)

            out["global_descriptors"] = X

            return out
    

def load_checkpoint(checkpoint_file, model):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint

    model_dict = model.state_dict()
    state_dict = {k : v for k, v in state_dict.items()}
    weight_dict = {k : v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    if len(weight_dict) == len(state_dict):
        print('All parameters are loaded')
    else:
        raise AssertionError("The model is not fully loaded.")

    model_dict.update(weight_dict)
    model.load_state_dict(model_dict)

    return checkpoint
