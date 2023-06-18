import os
import sys 
from pathlib import Path

patchnetvlad_path = Path(__file__).parent / '../../third_party/Patch-NetVLAD'
sys.path.append(str(patchnetvlad_path))

from patchnetvlad.models.patchnetvlad import PatchNetVLAD as _PatchNetVLAD
from patchnetvlad.models.local_matcher import calc_keypoint_centers_from_patches
from patchnetvlad.models.models_generic import get_backend, Flatten, L2Norm, get_pca_encoding
from download_models import download_all_models

import torch
from torch import nn
import logging
import torchvision.transforms as transforms
from PIL import Image
from .base_model import BaseModel


PATCHNETVLAD_DIR = "third_party/Patch-NetVLAD/patchnetvlad/pretrained_models"


class PatchNetVLAD(BaseModel):
    default_conf = {"vladv2": False,
                    "patch_sizes": '2,5,8',
                    "strides": '1,1,1',
                    "pretrained_patch_netvlad_model": "pittsburgh_WPCA",
                    "return_pca": True,
                    "num_pcs": 4096,
                    "resize_h": 480,
                    "resize_w": 640}

    required_inputs = ["image"]

    def _init(self, conf):
        use_pca = True if conf['num_pcs'] != 0 else False

        if use_pca:
            resume_ckpt = os.path.join(PATCHNETVLAD_DIR, conf["pretrained_patch_netvlad_model"] + str(conf['num_pcs']) + '.pth.tar')
        else:
            resume_ckpt = os.path.join(PATCHNETVLAD_DIR, conf["pretrained_patch_netvlad_model"] + '.pth.tar')

        if not os.path.isfile(resume_ckpt):
            download_all_models(ask_for_permission=False)

        logging.info("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)

        enc_dim, encoder = get_backend()

        self.nn_model = nn.Module()
        self.nn_model.add_module('encoder', encoder)

        num_cluster = int(checkpoint['state_dict']['pool.centroids'].shape[0])

        net_vlad = _PatchNetVLAD(num_clusters=num_cluster,
                                           dim=enc_dim, vladv2=conf['vladv2'], patch_sizes=conf['patch_sizes'],
                                           strides=conf['strides'])

        self.nn_model.add_module('pool', net_vlad)

        if use_pca:
            netvlad_output_dim = enc_dim * num_cluster
            pca_conv = nn.Conv2d(netvlad_output_dim, conf["num_pcs"], kernel_size=(1, 1), stride=1, padding=0)
            self.nn_model.add_module('WPCA', nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))


        logging.info("Restoring checkpoint")

        self.nn_model.load_state_dict(checkpoint['state_dict'])

        self.nn_model.eval()


    def forward(self, data):
        H, W = data["image"].shape[2:]

        image = data["image"]

        
        image_encoding = self.nn_model.encoder(image)

        vlad_local, vlad_global = self.nn_model.pool(image_encoding)

        if self.conf["return_pca"]:
            vlad_local_pca = []

            for this_iter, this_local in enumerate(vlad_local):
                this_local_pca = get_pca_encoding(self.nn_model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))).\
                            reshape(this_local.size(2), this_local.size(0), self.conf["num_pcs"]).permute(1, 2, 0)


                vlad_local_pca.append(this_local_pca)

            vlad_local = vlad_local_pca

        keypoints_list = []
        levels_list = []
        for i, this_local in enumerate(vlad_local):
            keypoints, _ = calc_keypoint_centers_from_patches(config=dict(imageresizeH=H, imageresizeW=W),
                                                              patch_size_h=self.nn_model.pool.patch_sizes[i],
                                                              patch_size_w=self.nn_model.pool.patch_sizes[i],
                                                              stride_h=self.nn_model.pool.strides[i],
                                                              stride_w=self.nn_model.pool.strides[i])

            keypoints = torch.FloatTensor(keypoints.T)[None]
            keypoints[:, :, 0] = keypoints[:, :, 0] 
            keypoints[:, :, 1] = keypoints[:, :, 1] 

            keypoints_list.append(keypoints)

            levels = torch.full((1, this_local.shape[-1]), i)
            levels_list.append(levels)

        vlad_local = torch.cat(vlad_local, -1)
        keypoints = torch.cat(keypoints_list, 1)
        levels = torch.cat(levels_list, -1)



        return {"descriptors": vlad_local, "keypoints": keypoints, "levels": levels}
    
    def get_preprocessing(self):
        if self.conf["resize_h"] > 0 and  self.conf["resize_w"] > 0:
            tv = input_transform_resize(resize=(self.conf["resize_h"], self.conf["resize_w"]))
        else:
            tv = input_transform_base()

        return tv


def input_transform_resize(resize=(480, 640)):
    return transforms.Compose([
            lambda path: Image.open(path).convert("RGB"),
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            lambda x: x[None]
        ])

def input_transform_base():
    return transforms.Compose([
            lambda path: Image.open(path).convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            lambda x: x[None]
        ])


