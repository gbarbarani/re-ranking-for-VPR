"""From https://github.com/cvg/Hierarchical-Localization"""

from pathlib import Path
import torch
from torchvision import transforms

from .base_model import BaseModel

from SuperGluePretrainedNetwork.models.superpoint import SuperPoint as _SuperPoint  
from PIL import Image


class SuperPoint(BaseModel):
    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024,
        'remove_borders': 4
    }
    required_inputs = ['image']
    detection_noise = 2.0

    def _init(self, conf):
        self.net = _SuperPoint(conf).eval()


    def forward(self, data):
        # Extract descriptors

        out = self.net(data)

        out['keypoints'] = out['keypoints'][0][None]
        out['scores'] = out['scores'][0][None]
        out['descriptors'] = out['descriptors'][0][None]

        return out

    def get_preprocessing(self):
        return transforms.Compose([
             lambda path: Image.open(path).convert("RGB"),
            transforms.ToTensor(),
            transforms.Grayscale(),
            lambda x: x[None]
        ])