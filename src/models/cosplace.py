import torch
from torchvision import transforms 
from .base_model import BaseModel
from PIL import Image


class CosPlace(BaseModel):
    default_conf = {
        'backbone': 'ResNet50',
        'fc_output_dim' : 512
    }
    required_inputs = ['image']
    def _init(self, conf):
        self.net = torch.hub.load(
            'gmberton/CosPlace',
            'get_trained_model',
            backbone=conf['backbone'],
            fc_output_dim=conf['fc_output_dim']
        ).eval()

        self.collate_fn = None
        self.output_dim = conf["fc_output_dim"]

    def forward(self, data):
        desc = self.net(data['image'])
        return {
            'global_descriptors': desc,
        }
    
    def get_preprocessing(self):
        return transforms.Compose([lambda path: Image.open(path).convert("RGB"),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
