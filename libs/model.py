import numpy as np
import sklearn
import sklearn.mixture
from libs.common import fv_type, model_type


class Model3D:
    def __init__(self):
        self.model = {}

    def fit_from_data(self, data3d, seeds, unique_cls):
        clx, classelected = self._features_from_data(data3d, seeds, unique_cls)
        clx = np.asarray(clx)
        cla = np.asarray(classelected)
        for cli in np.unique(cla):
            selection = cla == cli
            clxsel = clx[np.nonzero(selection)[0]]
            if model_type == 'gmm_same':
                self.model[cli] = sklearn.mixture.GaussianMixture()
                self.model[cli].fit(clxsel)

    def _features_from_data(self, data3d, seeds, unique_cls=None):
        fv = []
        if fv_type == 'intensity':
            fv = data3d.reshape(-1, 1)
            if seeds is not None:
                sd = seeds.reshape(-1, 1)
                selection = np.in1d(sd, unique_cls)
                fv = fv[selection]
                sd = sd[selection]
                return fv, sd
        return fv

    def linkelihood_from_data(self, data3d, value):
        fv = self._features_from_data(data3d, None)
        px = self.model[value].score_samples(fv)
        return px.reshape(data3d.shape)
