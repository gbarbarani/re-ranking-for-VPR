import numpy as np
from .base_model import BaseModel
import torch


class MultiScaleRAPID(BaseModel):
    default_conf = {"rapid_configs": {},
                    "num_levels": 3, "weights": [0.45, 0.15, 0.4]}

    required_inputs = ["image0", "image1",
                       "keypoints0", "keypoints1",
                       "descriptors0", "descriptors1",
                       "levels0", "levels1"]

    def _init(self, **conf):
        self.sorers = []

        if isinstance(self.conf["rapid_configs"], dict):
            self.conf["rapid_configs"] = [self.conf["rapid_configs"]] * self.conf["num_levels"]

        for i in range(self.conf["num_levels"]):
            self.sorers.append(RAPID(**self.conf["rapid_configs"][i]))




    def _forward(self, data):
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
             w_scores.append(out["score"] * self.conf["weights"][i])



         s_score = sum(w_scores)

         return {"score": s_score}
    


class RAPID(BaseModel):
    required_inputs = ["image0", "image1",
                       "keypoints0", "keypoints1",
                       "descriptors0", "descriptors1"]

    def _init(self, **conf):
        return

    def _forward(self, data):
        qfeat = data["descriptors0"][0]
        dbfeat = data["descriptors1"][0]

        indices_q = data["keypoints0"][0].T.cpu().numpy()
        indices_db = data["keypoints1"][0].T.cpu().numpy()


        fw_inds, bw_inds = torch_nn(qfeat, dbfeat)
        fw_inds = fw_inds.cpu().numpy()
        bw_inds = bw_inds.cpu().numpy()

        mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())

        if len(mutuals) == 0:
            s_score = 0.


        else:
            index_keypoints = indices_db[:, mutuals]
            query_keypoints = indices_q[:, fw_inds[mutuals]]

            spatial_dist = index_keypoints - query_keypoints # manhattan distance works reasonably well and is fast
            mean_spatial_dist = np.mean(spatial_dist, axis=1)

            # residual between a spatial distance and the mean spatial distance. Smaller is better
            s_dists_x = spatial_dist[0, :] - mean_spatial_dist[0]
            s_dists_y = spatial_dist[1, :] - mean_spatial_dist[1]
            s_dists_x = np.absolute(s_dists_x)
            s_dists_y = np.absolute(s_dists_y)

            # anchor to the maximum x and y axis index for the patch "feature space"
            xmax = max([np.max(indices_q[0, :]), np.max(indices_db[0, :])])
            ymax = max([np.max(indices_q[1, :]), np.max(indices_db[1, :])])

            # find second-order residual, by comparing the first residual to the respective anchors
            # after this step, larger is now better
            # add non-linearity to the system to excessively penalise deviations from the mean
            s_score = (xmax - s_dists_x)**2 + (ymax - s_dists_y)**2
            s_score = - s_score.sum()/qfeat.shape[1]

        return {"score": s_score}


def torch_nn(x, y):
    mul = torch.matmul(x.T, y)

    dist = 2 - 2 * mul

    fw_inds = torch.argmin(dist, 0)
    bw_inds = torch.argmin(dist, 1)

    return fw_inds, bw_inds
