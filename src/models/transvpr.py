import sys
import torch
import os
from pathlib import Path

from .base_model import BaseModel

from torchvision import transforms
from PIL import Image

import sys 
transvpr_path = Path(__file__).parent / '../../third_party/TransVPR-model-implementation'
sys.path.append(str(transvpr_path))

from feature_extractor import Extractor_base
from blocks import POOL


import numpy as np
import math
import torch.nn.functional as F


def get_keypoints(img_size):
    # flaten by x 
    H,W = img_size
    patch_size = 2**4
    N_h = H//patch_size
    N_w = W//patch_size
    keypoints = np.zeros((2, N_h*N_w), dtype=int) #(x,y)
    keypoints[0] = np.tile(np.linspace(patch_size//2, W-patch_size//2, N_w, 
                                       dtype=int), N_h)
    keypoints[1] = np.repeat(np.linspace(patch_size//2, H-patch_size//2, N_h,
                                         dtype=int), N_w)
    return np.transpose(keypoints)

def max_min_norm_tensor(mat):
    v_min,_ = torch.min(mat, -1, keepdims=True)
    v_max,_ = torch.max(mat, -1, keepdims=True)
    mat = (mat-v_min) / (v_max-v_min)
    return mat

def get_model(ckpt):
    model = Extractor_base()
    pool = POOL(model.embedding_dim)
    model.add_module('pool', pool)
    
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint)

    return model.eval()
    

class RescaleAndCentralCrop():
    def __init__(self, resize):
        self.resize = resize
        self.crop = transforms.CenterCrop(resize)

    def __call__(self, tensor_image):
        h, w = tensor_image.shape[-2:]
        resize_ratio = max([self.resize[0]/h, self.resize[1]/w, 1.])
        if resize_ratio > 1.:
            h_new = math.ceil(h * resize_ratio)
            w_new = math.ceil(w * resize_ratio)

            #tensor_image = F.interpolate(tensor_image, (h_new,w_new), mode='bilinear', align_corners=False)
            tensor_image = transforms.Resize((h_new, w_new))(tensor_image)
        
        return self.crop(tensor_image)
    



        
def get_preprocessing(mode, resize=(480, 640), central_crop=False):
    if central_crop:
        resize_transform = RescaleAndCentralCrop(resize)
    else:
        resize_transform = transforms.Resize(resize)

    if mode == "reranking":
        return transforms.Compose([
            lambda path: Image.open(path).convert("RGB"),
            transforms.ToTensor(),
            resize_transform,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            lambda x: x[None]])
    elif mode == "global":
        return transforms.Compose([
            lambda path: Image.open(path).convert("RGB"),
            transforms.ToTensor(),
            resize_transform,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class TransVPR(BaseModel):
    default_conf = {"resize": [480,640],
                    "central_crop": False,
                    "threshold": 0.02,
                    "layer": 3,
                    "pretrained_path": "third_party/TransVPR-model-implementation/TransVPR_Pitts30k.pth",
                    "mode": "reranking"}

    def _init(self, conf):    
        self.model = get_model(conf["pretrained_path"])
            
        self.collate_fn = None
        self.output_dim = 256

    def get_preprocessing(self):
        return get_preprocessing(self.conf["mode"], self.conf["resize"], self.conf["central_crop"])

    def forward(self, data):
        if self.conf["mode"] == "reranking":
            input = data['image']
            feature = self.model(input)
            encoding, mask = self.model.pool(feature)

        

            feature = feature[:, self.conf["layer"],1:, :]

            mask = mask.detach()
            
            mask = torch.sum(max_min_norm_tensor(mask), 1)
            mask = max_min_norm_tensor(mask)
            mask = (mask[0] > self.conf["threshold"])

            feature = feature[:, mask]
            feature = feature[0].t()[None]
        
            keypoints = get_keypoints(input.shape[-2:])
            keypoints = torch.from_numpy(keypoints)[None].to(mask.device)
            keypoints = keypoints[:, mask]
        
            return {
                'keypoints': keypoints,
                'descriptors': feature,
                'global_descriptors': encoding,
            }
        
        elif self.conf["mode"] == "global":
            
            input = data['image']
            feature = self.model(input)
            encoding, mask = self.model.pool(feature)

            return {"global_descriptors": encoding}

