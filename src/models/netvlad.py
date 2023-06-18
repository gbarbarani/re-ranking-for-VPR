from .base_model import BaseModel
from ...third_party.Patch_NetVLAD.patchnetvlad.models.netvlad import NetVLAD as _NetVLAD
from ...third_party.Patch_NetVLAD.patchnetvlad.models.models_generic import get_backend, Flatten, L2Norm, get_pca_encoding
from ...third_party.Patch_NetVLAD.download_models import download_all_models


import os
import torch
from torch import nn
import logging
import torchvision.transforms as transforms
from PIL import Image

PATCHNETVLAD_DIR = "third_party/Patch-NetVLAD/patchnetvlad/pretrained_models"


class NetVLAD(BaseModel):
    default_conf = {"vladv2": False,
                    "pretrained_netvlad_model": "pittsburgh_WPCA", #"mapillary_WPCA",
                    "return_pca": True,
                    "num_pcs": 4096,
                    "resize_h": 0,
                    "resize_w": 0}

    required_inputs = ["image"]

    def _init(self, **conf):
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

        net_vlad = _NetVLAD(num_clusters=num_cluster, dim=enc_dim, vladv2=conf['vladv2'])

        self.nn_model.add_module('pool', net_vlad)

        if use_pca:
            netvlad_output_dim = enc_dim * num_cluster
            pca_conv = nn.Conv2d(netvlad_output_dim, conf["num_pcs"], kernel_size=(1, 1), stride=1, padding=0)
            self.nn_model.add_module('WPCA', nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))


        logging.info("Restoring checkpoint")

        self.nn_model.load_state_dict(checkpoint['state_dict'])

        self.nn_model.eval()

    def _forward(self, data):
        
        image = data["image"]

        
        image_encoding = self.nn_model.encoder(image)

        vlad = self.nn_model.pool(image_encoding)
        if self.conf["return_pca"]:
            vlad = get_pca_encoding(self.nn_model, vlad)


        return {"global_descriptors": vlad}

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
        ])

def input_transform_base():
    return transforms.Compose([
            lambda path: Image.open(path).convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
