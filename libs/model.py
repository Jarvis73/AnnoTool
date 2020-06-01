import numpy as np
import os.path as op

import sklearn
import sklearn.mixture
from libs.common import default_model_params, gmm_cvtype_bad, gmm_cvtype


class Model3D:
    def __init__(self, model_params):
        self.mdl = {}
        self.model_params = default_model_params.copy()
        self.model_params.update({'adaptation': 'retrain'})
        self.model_params.update(model_params)

    def fit(self, clx, cla):
        if np.isscalar(cla):
            self._fit_one_class(clx, cla)
        else:
            cla = np.asarray(cla)
            clx = np.asarray(clx)
            for cli in np.unique(cla):
                selection = cla == cli
                clxsel = clx[np.nonzero(selection)[0]]
                self._fit_one_class(clxsel, cli)

    def _fit_one_class(self, clx, cl):
        if self.model_params["adaptation"] == "original_data":
            if cl in self.mdl.keys():
                return
        if self.model_params["type"] == "gmm_same":
            gmm_params = self.model_params["params"]
            self.mdl[cl] = sklearn.mixture.GaussianMixture(**gmm_params)
            self.mdl[cl].fit(clx)

    def fit_from_image(self, data, voxel_size, seeds, unique_cls):
        fvs, clsselected = self._features_from_image(data, voxel_size, seeds, unique_cls)
        self.fit(fvs, clsselected)

    def _features_from_image(self, data, voxel_size, seeds=None, unique_cls=None):
        fv_type = self.model_params["fv_type"]
        fv = []
        if fv_type == 'intensity':
            fv = data.reshape(-1, 1)
            if seeds is not None:
                sd = seeds.reshape(-1, 1)
                selection = np.in1d(sd, unique_cls)
                fv = fv[selection]
                sd = sd[selection]
                return fv, sd
        return fv
        # todo intensity and blur

    def likelihood_from_image(self, data, voxel_size, cl):
        sha = data.shape

        likel = self._likelihood(self._features_from_image(data, voxel_size), cl)
        return likel.reshape(sha)

    def _likelihood(self, x, cl):
        if self.model_params["type"] == "gmm_same":
            px = self.mdl[cl].score_samples(x)
        elif self.model_params["type"] == "kernel":
            px = self.mdl[cl].score_samples(x)
        elif self.model_params["type"] == "gaussian_kde":
            px = np.log(self.mdl[cl](x))
        elif self.model_params["type"] == "dpgmm":
            px = self.mdl[cl].score_samples(x * 0.01)
        elif self.model_params["type"] == "stored":
            px = self.mdl[cl].score(x)
        return px
