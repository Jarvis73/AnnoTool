import sys
import pygco
import numpy as np
from PyQt5.QtWidgets import QApplication
from libs.common import default_model_params
from libs.model import Model3D


class GraphCut3D:
    def __init__(self, img, voxel_size, model_params={}, seg_params={}):
        self.voxel_size = np.asarray(voxel_size)
        self.voxel_volume = np.prod(voxel_size)
        self._update_seg_params(seg_params, model_params)
        self.img = img
        self.segmentation = None
        self.seeds = np.zeros(self.img.shape, dtype=np.int8)

        self.interactivity_counter = 0
        self.stats = {"tlinks shape": [], "nlinks shape": []}
        self.apriori = None

    def _update_seg_params(self, seg_params, model_params):
        self.seg_params = {"method": "graphcut", "pairwise_alpha": 20, "use_boundary_penalties": False,
                           "boundary_penalties_sigma": 200, "boundary_penalties_weight": 30,
                           "return_only_object_with_seeds": False, "use_extra_features_for_training": False,
                           "use_apriori_if_available": True, "apriori_gamma": 0.1}
        if "model_params" in seg_params.keys():
            model_params = seg_params["model_params"]
        self.seg_params.update(seg_params)
        self.model_params = default_model_params.copy()
        self.model_params.update(model_params)
        self.mdl = Model3D(model_params=self.model_params)

    def set_seeds(self, seeds):
        self.seeds = seeds.astype("int8")
        self.voxels1 = self.img[self.seeds == 1]
        self.voxels2 = self.img[self.seeds == 2]

    def run(self):
        self._fit_model()
        if self.seg_params["method"].lower() in ("graphcut"):
            self._single_scale_gc_run()

    def _fit_model(self):
        if self.seg_params["use_extra_features_for_training"]:
            self.mdl.fit(self.voxels1, 1)
            self.mdl.fit(self.voxels2, 2)
        else:
            self.mdl.fit_from_image(self.img, self.voxel_size, self.seeds, [1, 2])

    def _single_scale_gc_run(self):
        res_segm = self.__ssgc_prepare_data_and_run_computation()
        self.segmentation = res_segm.astype(np.int8)

    def __ssgc_prepare_data_and_run_computation(self, hard_constraints=True, area_weight=1):
        unariesalt = self.__create_tlinks(self.img, self.voxel_size, self.seeds, area_weight, hard_constraints)
        pairwise = -(np.eye(2) - 1)
        pairwise = (self.seg_params["pairwise_alpha"] * pairwise).astype(np.int32)
        self.iparams = {}
        if self.seg_params["use_boundary_penalties"]:
            sigma = self.seg_params["boundary_penalties_sigma"]
            boundary_penalties_fcn = lambda ax: self._boundary_penalties_array(axis=ax, sigma=sigma)
        else:
            boundary_penalties_fcn = None
        nlinks = self.__create_nlinks(self.img, boundary_penalties_fcn=boundary_penalties_fcn)
        self.stats["tlinks shape"].append(unariesalt.reshape(-1, 2).shape)
        self.stats["nlinks shape"].append(nlinks.shape)
        result_graph = pygco.cut_from_graph(nlinks, unariesalt.reshape(-1, 2), pairwise)
        result_labeling = result_graph.reshape(self.img.shape)
        return result_labeling

    def __set_hard_hard_constraints(self, tdata1, tdata2, seeds):
        seeds_mask = (seeds == 1) | (seeds == 3)
        tdata2[seeds_mask] = np.max(tdata2) + 1
        tdata1[seeds_mask] = 0
        seeds_mask = (seeds == 2) | (seeds == 4)
        tdata1[seeds_mask] = np.max(tdata1) + 1
        tdata2[seeds_mask] = 0
        return tdata1, tdata2

    def __limit(self, tdata1, min_limit=0, max_error=10, max_limit=20000):
        tdata1[tdata1 > max_limit] = max_limit
        tdata1[tdata1 < min_limit] = min_limit
        return tdata1

    def __create_tlinks(self, data, voxel_size, seeds, area_weight, hard_constraints, mul_mask=None, mul_val=None):
        tdata1, tdata2 = self.__similarity_for_tlinks_obj_bgr(data, voxel_size)
        if hard_constraints:
            tdata1, tdata2 = self.__set_hard_hard_constraints(tdata1, tdata2, seeds)
        tdata1 = self.__limit(tdata1)
        tdata2 = self.__limit(tdata2)
        unariesalt = (0 + (np.dstack(area_weight * [tdata1.reshape(-1, 1), tdata2.reshape(-1, 1)]).copy("C"))).astype(
            np.int32)
        unariesalt = self.__limit(unariesalt)
        return unariesalt

    def _prepare_edgs_arr_with_no_fcn(self, inds):
        if self.img.ndim == 3:
            edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
            edgs_arr = [edgx, edgy, edgz]
        elif self.img.ndim == 2:
            edgx = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
            edgy = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
            edgs_arr = [edgx, edgy]
        else:
            edgs_arr = None
        return edgs_arr

    def __create_nlinks(self, data, inds=None, boundary_penalties_fcn=None):
        if inds is None:
            inds = np.arange(data.size).reshape(data.shape)
        if boundary_penalties_fcn is None:
            edgs_arr = self._prepare_edgs_arr_with_no_fcn(inds)

        edges = np.vstack(edgs_arr).astype(np.int32)
        return edges

    def __similarity_for_tlinks_obj_bgr(self, data, voxel_size):
        tdata1 = (-(self.mdl.likelihood_from_image(data, voxel_size, 1))) * 10 # todo
        tdata2 = (-(self.mdl.likelihood_from_image(data, voxel_size, 2))) * 10
        dtype = np.int16
        if np.any(tdata1 > 32760):
            dtype = np.float32
        if np.any(tdata2 > 32760):
            dtype = np.float32

        if self.seg_params["use_apriori_if_available"] and self.apriori is not None:
            gamma = self.seg_params["apriori_gamma"]
            a1 = (-np.log(self.apriori * 0.998 + 0.001)) * 10
            a2 = (-np.log(0.999 - (self.apriori * 0.998))) * 10

            tdata1u = (((1 - gamma) * tdata1) + (gamma * a1)).astype(dtype)
            tdata2u = (((1 - gamma) * tdata2) + (gamma * a2)).astype(dtype)
            tdata1 = tdata1u
            tdata2 = tdata2u
            del tdata1u
            del tdata2u
            del a1
            del a2
        return tdata1, tdata2

