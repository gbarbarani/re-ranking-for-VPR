from .base_model import BaseModel
from kornia.feature import LoFTR as _LoFTR
from torchvision import transforms
from PIL import Image

class LoFTR(BaseModel):
    default_conf = {}
    required_inputs = ["image0", "image1"]

    def _init(self, **conf):
        self.matcher = _LoFTR(pretrained='outdoor')

    def _forward(self, data):
        out = self.matcher({"image0": data["image0"],
                            "image1": data["image1"]})

        out["score"] = - len(out["keypoints0"])

        return out
    
    def get_preprocessing(self):
        return transforms.Compose([
             lambda path: Image.open(path).convert("RGB"),
            transforms.ToTensor(),
            transforms.Grayscale(),
            lambda x: x[None]
        ])
