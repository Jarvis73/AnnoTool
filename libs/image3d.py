import cv2
import grpc
import numpy as np
import nibabel as nib
import qimage2ndarray as q2a
import skimage.measure as measure
from skimage import feature
from skimage._shared import utils as skutils
# import tensorflow as tf
# from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pathlib import Path
from collections import OrderedDict
from libs.graph_cut import GraphCut3D


class Image3d(object):
    """ For 3D gray image """

    def __init__(self, volume, meta=None, filePath=None):
        # self.raw = raw
        # self.volume = raw.copy()
        if volume is None and meta is None:
            raise ValueError("Both `volume` and `meta` are None.")
        self.volume = volume
        self.filePath = filePath
        self.meta = meta
        if volume is not None:
            self.shape = volume.shape
            self.low = self.volume.min()
            self.high = self.volume.max()
        else:
            self.shape = self.meta.shape
            self.low = 0
            self.high = 0
        self.axis = 0		# 0, 1, 2

        self.segCache = {}
        # if filePath:
        #     filePath = Path(filePath)
        #     # segPath = filePath.parent.parent / "Lasso_nii" / filePath.name.replace("volume", "segmentation")
        #     segPath = filePath.parent.parent / "temp" / filePath.name.replace("volume", "segmentation")
        #     if segPath.exists():
        #         _, self.segLabel = read_nii(segPath, out_dtype=np.uint8)
        #         self.segCache = np.clip(self.segLabel, 0, 1)

        self._alpha = 0.5
        self._color = np.array([0, 255, 0])
        self.contour = False
        self.backup = None
        self.guide_temp = None

    def __getitem__(self, i):
        return self.volume[i]

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color_):
        self._color = np.array(color_)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha_):
        self._alpha = max(min(alpha_, 1), 0)

    def astype(self, dtype):
        self.volume = self.volume.astype(dtype)

    def check(self, i, delta, axis=0):
        if i - delta < 0:
            return 0
        if i - delta >= self.shape[axis]:
            return self.shape[axis] - 1
        return i - delta

    def setIntensityClip(self, low=None, high=None):
        if low is not None and isinstance(low, (int, float)) and self.low != low:
            self.low = low
        if high is not None and isinstance(high, (int, float)) and self.high != high:
            self.high = high

    def at(self, i, axis=0, show=True):
        # We don't check index out of bounds
        if axis == 0:
            img = self.volume[i]
        elif axis == 1:
            img = self.volume[:, i]
        elif axis == 2:
            img = self.volume[:, :, i]

        if not show or len(self.segCache) == 0 or (isinstance(self.segCache, dict) and i not in self.segCache):
            return q2a.gray2qimage(img, normalize=(self.low, self.high))
        else:
            obj = self.segCache[i]
            if self.segCache[i].max() > 1:
                obj = np.clip(self.segCache[i], 0, 1)
            img = np.repeat(img[..., None], 3, axis=2)
            if not self.contour:
                px = np.where(obj)
                img[px[0], px[1]] = (1 - self._alpha) * img[px[0], px[1]] + self._alpha * self._color
            else:
                contours = measure.find_contours(obj, 0.5)
                for cont in contours:
                    cont = cont.astype(np.int16)
                    img[cont[:, 0], cont[:, 1]] = self._color
            return q2a.array2qimage(img, normalize=(self.low, self.high))

    def pixel(self, slice_idx, i, j, axis=0):
        if axis == 0:
            return self.volume[slice_idx, i, j]
        elif axis == 1:
            return self.volume[i, slice_idx, j]
        elif axis == 2:
            return self.volume[i, j, slice_idx]
        return 0

    def undo(self):
        if self.backup is not None:
            self.segCache = self.backup
            self.backup = None
            return True
        return False

    def seg_test(self, bbox, centers, stddevs):
        z1, y1, x1, z2, y2, x2 = bbox
        seg = np.zeros_like(self.volume, np.uint8)
        seg[z1:z2, y1:y2, x1:x2] = 1
        for c in centers['fg']:
            seg[c[0], c[1] - 15: c[1] + 15, c[2] - 15: c[2] + 15] = 0
        for c in centers['bg']:
            seg[c[0], c[1] - 7: c[1] + 7, c[2] - 7: c[2] + 7] = 0
        self.segCache = seg
        return 0

    def seg_gc(self, centers):
        gc = GraphCut3D(self.volume, [1, 1, 1])
        seeds = np.zeros_like(self.volume, np.uint8)
        fg = np.array(centers["fg"], np.int32).reshape(-1, 3)
        bg = np.array(centers["bg"], np.int32).reshape(-1, 3)
        if fg.shape[0] > 0:
            seeds[fg[:, 0], fg[:, 1], fg[:, 2]] = 1
        if bg.shape[0] > 0:
            seeds[bg[:, 0], bg[:, 1], bg[:, 2]] = 1
        gc.set_seeds(seeds)
        gc.run()
        self.segCache = gc.segmentation
        return 0


class Header(object):
    def __init__(self, meta, format):
        self.meta = meta
        self.fmt = format

    @property
    def shape(self):
        if self.fmt in ["nii", "nii.gz"]:
            return [int(x) for x in self.meta.get_data_shape()[::-1]]


class DSKey(object):
    """ (Direction, Slice) <-> Key """

    @staticmethod
    def ds2key(di, sl):
        return di * 10000 + sl

    @staticmethod
    def key2ds(key):
        return key // 10000, key % 10000


class Item3d(object):
    def __init__(self, label, axis, slice_):
        self.label = label
        self.axis = axis
        self.slice = slice_


def computeDice(ref, pred, bbox):
    if not (isinstance(ref, np.ndarray) and isinstance(pred, np.ndarray) and ref.shape == pred.shape):
        print(ref.shape, pred.shape)
        return -1

    ref_choose = np.clip(ref, 0, 1)[bbox]
    pred_choose = pred[bbox]

    v1 = np.count_nonzero(ref_choose * pred_choose)
    v2 = np.count_nonzero(ref_choose) + np.count_nonzero(pred_choose)
    return round(2 * v1 / (v2 + 1e-8), 4)


def read3d(filePath, out_dtype=np.int16, only_header=False):
    if filePath.lower().endswith(('.nii', '.nii.gz')):
        hdr, volume = read_nii(filePath, out_dtype, only_header)
        return Image3d(volume, Header(hdr, "nii"), filePath)


def read_nii(file_name, out_dtype=np.int16, only_header=False):
    nib_vol = nib.load(str(file_name))
    vh = nib_vol.header
    if only_header:
        return vh, None
    affine = vh.get_best_affine()
    # assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    data = nib_vol.get_fdata().astype(out_dtype).transpose(*trans[::-1])

    if affine[0, trans[0]] > 0:                # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:                # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:                # Increase z from Interior to Superior
        data = np.flip(data, axis=0)
    return vh, data


def write_nii(data, header, out_path, out_dtype=np.int16, affine=None):
    if header is not None:
        affine = header.get_best_affine()
    assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    trans_bk = [np.argwhere(np.array(trans[::-1]) == i)[0][0] for i in range(3)]

    if affine[0, trans[0]] > 0:  # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:  # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:  # Increase z from Interior to Superior
        data = np.flip(data, axis=0)

    out_image = np.transpose(data, trans_bk).astype(out_dtype)
    if header is None and affine is not None:
        out = nib.Nifti1Image(out_image, affine=affine)
    else:
        out = nib.Nifti1Image(out_image, affine=None, header=header)
    nib.save(out, str(out_path))


def createEllipse(rect, shape):
    x, y, w, h = rect
    cy, cx = h // 2, w // 2
    xrng = np.arange(w)
    yrng = np.arange(h)
    coord = np.stack(np.meshgrid(yrng, xrng, indexing="ij"), axis=-1)
    ellipse_mask = np.zeros(shape, dtype=np.uint8)
    ellipse_mask[y:y + h, x:x + w] = np.sum(((coord - [cy, cx]) / [cy, cx]) ** 2, axis=-1) <= 1
    return ellipse_mask
