"""From https://github.com/cvg/Hierarchical-Localization"""

from .base_model import BaseModel
from SuperGluePretrainedNetwork.models.superglue import SuperGlue as _SuperGlue


class SuperGlue(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }
    required_inputs = [
        'image0', 'keypoints0', 'scores0', 'descriptors0',
        'image1', 'keypoints1', 'scores1', 'descriptors1',
    ]

    def _init(self, conf):
        self.net = _SuperGlue(conf).eval()

    def forward(self, data):
        out = self.net(data)

        score = - (out["matches0"] != -1).sum().detach().cpu().numpy()

        out["score"] = score

        return out
