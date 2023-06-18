import numpy as np
import cv2
from .base_model import BaseModel
import torch


class MultiScaleRANSAC(BaseModel):
    default_conf = {"ransac_configs": {"ransacReprojThreshold": 24.,  "ransacIter": 2000},
                    "num_levels": 3, "weights": [0.45, 0.15, 0.4]}

    required_inputs = ["image0", "image1",
                       "keypoints0", "keypoints1",
                       "descriptors0", "descriptors1",
                       "levels0", "levels1"]

    def _init(self, conf):
        self.sorers = []

        if isinstance(conf["ransac_configs"], dict):
            conf["ransac_configs"] = [conf["ransac_configs"]] * conf["num_levels"]

        for i in range(self.conf["num_levels"]):
            self.sorers.append(RANSAC(**conf["ransac_configs"][i]))




    def forward(self, data):
         keypoints0_list = []
         keypoints1_list = []
         levels = []
         w_scores = []

         for i in range(self.conf["num_levels"]):
             _data = {"image0": data["image0"],
                      "image1": data["image1"]}

             mask0 = (data["levels0"][0]==i)
             mask1 = (data["levels1"][0]==i)

             _data["keypoints0"] = data["keypoints0"][:, mask0]
             _data["keypoints1"] = data["keypoints1"][:, mask1]

             _data["descriptors0"] = data["descriptors0"][:, :, mask0]
             _data["descriptors1"] = data["descriptors1"][:, :, mask1]

             out = self.sorers[i](_data)
             keypoints0_list.append(out["keypoints0"])
             keypoints1_list.append(out["keypoints1"])
             w_scores.append(out["score"] * self.conf["weights"][i])

             levels += [i] * len(out["keypoints0"])



         s_score = sum(w_scores)
         inlier_query_keypoints = np.concatenate(keypoints0_list, 0)
         inlier_index_keypoints = np.concatenate(keypoints1_list, 0)
         levels = np.array(levels)


         return {"score": s_score, "keypoints0": inlier_query_keypoints, "keypoints1": inlier_index_keypoints,
                 "levels": levels}


class RANSAC(BaseModel):
    default_conf = {"ransacReprojThreshold": 24., 
                    "ransacIter": 2000}
    required_inputs = ["image0", "image1",
                       "keypoints0", "keypoints1",
                       "descriptors0", "descriptors1"]

    def _init(self, conf):
        return

    def forward(self, data):
        qfeat = data["descriptors0"][0]
        dbfeat = data["descriptors1"][0]

        keypoints_q = data["keypoints0"][0].T.cpu().numpy()
        keypoints_db = data["keypoints1"][0].T.cpu().numpy()

        fw_inds, bw_inds = torch_nn(qfeat, dbfeat)
        fw_inds = fw_inds.cpu().numpy()
        bw_inds = bw_inds.cpu().numpy()

        mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

        if len(mutuals) == 0:
            s_score = 0.
            inlier_query_keypoints = []
            inlier_index_keypoints = []

        elif len(mutuals) < 4:
            inlier_index_keypoints = np.transpose(keypoints_db[:, mutuals])
            inlier_query_keypoints = np.transpose(keypoints_q[:, fw_inds[mutuals]])

            inlier_count = inlier_index_keypoints.shape[0]
            s_score = -inlier_count / qfeat.shape[1]

        else:
            index_keypoints = keypoints_db[:, mutuals]
            query_keypoints = keypoints_q[:, fw_inds[mutuals]]

            index_keypoints = np.transpose(index_keypoints)
            query_keypoints = np.transpose(query_keypoints)
             

            _, mask = cv2.findHomography(index_keypoints, query_keypoints, cv2.FM_RANSAC, 
                                         ransacReprojThreshold=self.conf["ransacReprojThreshold"], 
                                         maxIters=self.conf["ransacIter"])

            good_matches = mask.ravel() == 1

            inlier_index_keypoints = index_keypoints[good_matches]
            inlier_query_keypoints = query_keypoints[good_matches]

            inlier_count = inlier_index_keypoints.shape[0]
            s_score = -inlier_count / qfeat.shape[1]

        return {"score": s_score, "keypoints0": inlier_query_keypoints, "keypoints1": inlier_index_keypoints}


def torch_nn(x, y):
    mul = torch.matmul(x.T, y)

    dist = 2 - 2 * mul

    fw_inds = torch.argmin(dist, 0)
    bw_inds = torch.argmin(dist, 1)

    return fw_inds, bw_inds
