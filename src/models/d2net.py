"""From https://github.com/cvg/Hierarchical-Localization"""

import subprocess
import torch
import os
from pathlib import Path
from .base_model import BaseModel

import sys 
d2net_path = Path(__file__).parent / '../../third_party/d2-net'
sys.path.append(str(d2net_path))
from lib.model_test import D2Net as _D2Net
from lib.pyramid import process_multiscale

#from d2_net.lib.model_test import D2Net as _D2Net
#from d2_net.lib.pyramid import process_multiscale
from torchvision import transforms
from PIL import Image


class CaffeNormalization():
    def __call__(self, image):
        image = image.flip(1)  # RGB -> BGR
        norm = image.new_tensor([103.939, 116.779, 123.68])
        image = (image * 255 - norm.view(3, 1, 1))  # caffe normalization

        return image
    

class D2Net(BaseModel):
    default_conf = {
        'model_name': 'd2_tf.pth',
        'checkpoint_dir': 'third_party/d2-net/',
        'use_relu': True,
        'multiscale': True,
    }
    required_inputs = ['image']

    def _init(self, conf):
        conf['checkpoint_dir'] = Path(conf['checkpoint_dir'])

        model_file = conf['checkpoint_dir'] / conf['model_name']
        if not model_file.exists():
            model_file.parent.mkdir(exist_ok=True)
            cmd = ['wget', 'https://dsmn.ml/files/d2-net/'+conf['model_name'],
                   '-O', str(model_file)]
            subprocess.run(cmd, check=True)

        self.net = _D2Net(
            model_file=model_file,
            use_relu=conf['use_relu'],
            use_cuda=False).eval()

    def forward(self, data):
        image = data['image']

        with torch.no_grad():
            if self.conf['multiscale']:
                keypoints, scores, descriptors = process_multiscale(
                    image, self.net)
            else:
                keypoints, scores, descriptors = process_multiscale(
                    image, self.net, scales=[1])
            keypoints = keypoints[:, [1, 0]]  # (x, y) and remove the scale

        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors.T)[None],
        }

    def get_preprocessing(self):
        return transforms.Compose([lambda path: Image.open(path).convert("RGB"),
                                   transforms.ToTensor(),
                                   CaffeNormalization(),
                                   lambda x: x[None]])

