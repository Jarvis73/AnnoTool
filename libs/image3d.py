import os
import cv2
import grpc
import numpy as np
import nibabel as nib
import qimage2ndarray as q2a
import skimage.measure as measure
from skimage import feature
from skimage._shared import utils as skutils
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pathlib import Path
from collections import OrderedDict
import subprocess

from libs import image_np_ops
try:
    from libs.graph_cut import graph_cut3d
except ImportError as e:
    print(e)

patch_cache = {}
max_height_, max_width_ = 960, 320
# max_height_, max_width_ = 256, 256


class Image3d(object):
    """ For 3D gray image """

    def __init__(self, volume, meta=None, filePath=None, label=None):
        # self.raw = raw
        # self.volume = raw.copy()
        if volume is None and meta is None:
            raise ValueError("Both `volume` and `meta` are None.")
        self.volume = volume
        self.label = label
        self.filePath = filePath
        self.meta = meta
        self.unit = meta.unit
        if volume is not None:
            self.shape = volume.shape
            self.low = self.volume.min()
            self.high = self.volume.max()
        else:
            self.shape = self.meta.shape
            self.low = 0
            self.high = 0
        self.axis = 0  # 0, 1, 2

        self.segCache = {}
        # if filePath:
        #     filePath = Path(filePath)
        #     # segPath = filePath.parent.parent / "Lasso_nii" / filePath.name.replace("volume", "segmentation")
        #     segPath = filePath.parent.parent / "temp" / filePath.name.replace("volume", "segmentation")
        #     if segPath.exists():
        #         _, self.segLabel = read_nii(segPath, out_dtype=np.uint8)
        #         self.segCache = np.clip(self.segLabel, 0, 1)

        self._alpha = 0.5
        self._color = np.array([255, 255, 0])
        self._colorGT = np.array([255, 0, 0])
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
    def colorGT(self):
        return self._colorGT

    @colorGT.setter
    def colorGT(self, color_):
        self._colorGT = np.array(color_)

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

    def at(self, i, axis=0, showSeg=True, showLab=False):
        # We don't check index out of bounds
        if axis == 0:
            img = self.volume[i]
        elif axis == 1:
            img = self.volume[:, i]
        elif axis == 2:
            img = self.volume[:, :, i]

        img = np.repeat(np.clip(img[..., None], self.low, self.high), 3, axis=2)
        img = ((img - self.low) / (self.high - self.low) * 255).astype(np.uint8)
        if showLab:
            obj = self.label[i]
            if not self.contour:
                px = np.where(obj)
                img[px[0], px[1]] = (1 - self._alpha) * img[px[0], px[1]] + self._alpha * self._colorGT
            else:
                contours = measure.find_contours(obj, 0.5)
                for cont in contours:
                    cont = cont.astype(np.int16)
                    img[cont[:, 0], cont[:, 1]] = self._colorGT
        if showSeg and len(self.segCache) != 0:
            obj = self.segCache[i]
            if self.segCache[i].max() > 1:
                obj = np.clip(self.segCache[i], 0, 1)
            if not self.contour:
                px = np.where(obj)
                img[px[0], px[1]] = (1 - self._alpha) * img[px[0], px[1]] + self._alpha * self._color
            else:
                contours = measure.find_contours(obj, 0.5)
                for cont in contours:
                    cont = cont.astype(np.int16)
                    img[cont[:, 0], cont[:, 1]] = self._color
        return q2a.array2qimage(img)

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

    def preprocess(self, bbox, centers, stddevs, guide_type="exp"):
        z1, y1, x1, z2, y2, x2 = bbox
        key = str(bbox) + str(self.filePath)
        # Image
        if key in patch_cache:
            image, zoom_scale, self.slices = patch_cache[key]
        else:
            patch = self.volume[z1:z2, y1:y2, x1:x2]
            if patch.min() < 0:
                patch[patch < 0] = 0
            patch = image_np_ops.z_score(patch)
            p1, p2, p3 = 0, 0, 0
            if y2 - y1 > max_height_ or x2 - x1 > max_width_:
                zoom_scale = np.array([1, max_height_ / patch.shape[1], max_width_ / patch.shape[2]])
                patch = ndi.zoom(patch, zoom_scale, order=1)
            else:
                zoom_scale = None
                if patch.shape[1] % 16 != 0:
                    p2 = (patch.shape[1] + 15) // 16 * 16 - patch.shape[1]
                if patch.shape[2] % 16 != 0:
                    p3 = (patch.shape[2] + 15) // 16 * 16 - patch.shape[2]
            if patch.shape[0] % 2 != 0:
                p1 = 1
            image = np.pad(patch, ((0, p1), (0, p2), (0, p3))).astype(np.float32)
            self.slices = (slice(-p1) if p1 > 0 else slice(None),
                           slice(-p2) if p2 > 0 else slice(None),
                           slice(-p3) if p3 > 0 else slice(None))
            
            patch_cache.clear()
            patch_cache[key] = [image.copy(), zoom_scale, self.slices]

        if y2 - y1 > max_height_ or x2 - x1 > max_width_:
            fg_pts = (np.array(centers['fg'], np.int32).reshape(-1, 3) - [z1, y1, x1]) * zoom_scale
            bg_pts = (np.array(centers['bg'], np.int32).reshape(-1, 3) - [z1, y1, x1]) * zoom_scale
            fg_std = np.array(stddevs['fg']).reshape(-1, 3) * zoom_scale
            bg_std = np.array(stddevs['bg']).reshape(-1, 3) * zoom_scale
        else:
            fg_pts = np.array(centers['fg'], np.int32).reshape(-1, 3) - [z1, y1, x1]
            bg_pts = np.array(centers['bg'], np.int32).reshape(-1, 3) - [z1, y1, x1]
            fg_std = np.array(stddevs['fg']).reshape(-1, 3)
            bg_std = np.array(stddevs['bg']).reshape(-1, 3)

        if guide_type == "exp":
            fg_gd = image_np_ops.gen_guide_nd_v2(image.shape, fg_pts, fg_std)
            bg_gd = image_np_ops.gen_guide_nd_v2(image.shape, bg_pts, bg_std) * 1.5
        elif guide_type == "euc":
            fg_gd = image_np_ops.gen_guide_nd_v2(image.shape, fg_pts, fg_std, euclidean=True)
            bg_gd = image_np_ops.gen_guide_nd_v2(image.shape, bg_pts, bg_std, euclidean=True) * 1.5
        elif guide_type == "geo":
            fg_gd = image_np_ops.gen_guide_geo_nd(image, fg_pts, lamb=1.0, iter_=2)
            bg_gd = image_np_ops.gen_guide_geo_nd(image, bg_pts, lamb=1.0, iter_=2)
        guide = np.stack((fg_gd, bg_gd), axis=-1).astype(np.float32)

        return image[None, ..., None], guide[None]

    def run_tf_serving(self, image, guide, name):
        host = "localhost"
        port = 8500
        options = [('grpc.max_send_message_length', 50 * 1536 * 512 * 4),
                   ('grpc.max_receive_message_length', 50 * 1536 * 512 * 4)]
        channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port), options=options)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = name
        request.model_spec.signature_name = "serving_default"
        request.inputs['image'].CopyFrom(tf.make_tensor_proto(image))
        request.inputs['guide'].CopyFrom(tf.make_tensor_proto(guide))

        try:
            result = stub.Predict(request)
        except Exception as e:
            print(e)
            result = None

        # Reference:
        # How to access nested values
        # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
        # logits = np.array(result.outputs["output_bytes2"].float_val).reshape(height, width, 2)
        if result is not None:
            output = np.array(result.outputs["output_0"].float_val).reshape(*guide.shape)
            output = output[0]
            return output
        else:
            return None

    def postprocess(self, mask, pts):
        struct = ndi.generate_binary_structure(3, 1)
        labeled, n_objs = ndi.label(mask)
        slices = ndi.find_objects(labeled)
        for i, sli in enumerate(slices, start=1):
            patch = labeled[sli] == i
            d, h, w = patch.shape
            if np.count_nonzero(patch) < 10:
                labeled[sli] -= patch * i
            # if np.count_nonzero(patch) < 40:
            #     has_pt = False
            #     for z, y, x in pts:
            #         if 0 <= z - sli[0].start < d and 0 <= y - sli[1].start < h and 0 <= x - sli[2].start < w:
            #             if patch[z - sli[0].start, y - sli[1].start, x - sli[2].start]:
            #                 has_pt = True
            #                 break
            #     if not has_pt:
            #         labeled[sli] -= patch * i

        labeled = np.clip(labeled, 0, 1)
        # Fill hole
        struct = ndi.generate_binary_structure(2, 1)
        labeled[pts[0, 0]] = ndi.morphology.binary_fill_holes(labeled[pts[0, 0]], struct)
        return labeled

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

    def seg_din(self, bbox, centers, stddevs, name, guide_type="exp"):
        z1, y1, x1, z2, y2, x2 = bbox
        if z1 is None:
            s = self.volume.shape
            bbox = [0, 0, 0, s[0], s[1], s[2]]
            z1, y1, x1, z2, y2, x2 = bbox
        image, guide = self.preprocess(bbox, centers, stddevs, guide_type)
        print(image.shape, guide.shape)
        logits = self.run_tf_serving(image, guide, name=name)
        if logits is None:
            return 1
        predict = np.argmax(logits, axis=-1)
        predict = predict[self.slices]
        if y2 - y1 > max_height_ or x2 - x1 > max_width_:
            zoom_scale = np.array([1, (y2 - y1) / max_height_, (x2 - x1) / max_width_])
            predict = ndi.zoom(predict, zoom_scale, order=0)
        seg = np.zeros_like(self.volume, np.uint8)
        seg[z1:z2, y1:y2, x1:x2] = predict
        self.segCache = self.postprocess(seg, np.array(centers["fg"]).reshape(-1, 3))
        return 0

    def seg_RW(self, bbox, centers):
        fg_pts = np.array(centers["fg"], np.int32).reshape(-1, 3)
        bg_pts = np.array(centers["bg"], np.int32).reshape(-1, 3)
        if fg_pts.shape[0] == 0 or bg_pts.shape[0] == 0:
            return 1
        z1, y1, x1, z2, y2, x2 = bbox
        if z1 is None:
            s = self.volume.shape
            bbox = [0, 0, 0, s[0], s[1], s[2]]
            z1, y1, x1, z2, y2, x2 = bbox
        fg_pts -= [z1, y1, x1]
        bg_pts -= [z1, y1, x1]
        patch = self.volume[z1:z2, y1:y2, x1:x2]
        tmp_dir = Path(__file__).parent.parent / "temp"
        infile = tmp_dir / "data.bin"
        size = np.zeros(3, dtype='int16')
        size[:] = patch.shape[:]
        # Convert data to binary
        with infile.open('wb') as inwrite:
            size.tofile(inwrite)
            patch.tofile(inwrite)
        # Create point.txt
        point_file = tmp_dir / "points.txt"
        with point_file.open("w") as f:
            f.write(f"{0},{patch.shape[0]},{0},{patch.shape[1]},{0},{patch.shape[2]}\n")
            for pt in fg_pts:
                f.write(f"1,{pt[0]},{pt[1]},{pt[2]}\n")
            for pt in bg_pts:
                f.write(f"0,{pt[0]},{pt[1]},{pt[2]}\n")
            f.write("end")
        # Perform Random Walker Segmentation
        # subprocess.call(["./tools/RandomWalk-3D.exe", "./temp/data.bin", "./temp/out.bin", "./temp/points.txt"])
        if not os.path.exists(".\\tools\\RandomWalk-3D.exe"):
            return 1
        os.system(".\\tools\\RandomWalk-3D.exe .\\temp\\data.bin .\\temp\\out.bin .\\temp\\points.txt")
        outfile = tmp_dir / "out.bin"
        with outfile.open('rb') as binread:
            shape = np.fromfile(binread, dtype=np.int16, count=3, sep='')
            ctdata = np.fromfile(binread, dtype=np.int16, count=-1, sep='')
            ctdata = ctdata.reshape(shape)
        self.segCache = np.zeros_like(self.volume, np.uint8)
        self.segCache[z1:z2, y1:y2, x1:x2] = ctdata
        os.remove(outfile)
        return 0

    def seg_GraphCut(self, bbox, centers):
        z1, y1, x1, z2, y2, x2 = bbox
        if len(centers['fg']) <= 1 or len(centers['bg']) <= 1:
            return 1
        seed = np.zeros_like(self.volume, np.uint8)
        seg = np.zeros_like(self.volume, np.uint8)
        for key, values in centers.items():
            _type = 1 if key == 'fg' else 2
            for point in values:
                seed[point[0]][point[1]][point[2]] = _type
            # print(1)
        box_volume = self.volume[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1]
        box_seed = seed[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1]
        box_seg = 1 - graph_cut3d(box_volume, box_seed)
        seg[z1:z2 + 1, y1:y2 + 1, x1:x2 + 1] = box_seg
        self.segCache = seg
        return 0


class Header(object):
    def __init__(self, meta, format):
        self.meta = meta
        self.fmt = format
        self.unit = np.prod(meta['pixdim'][1:4]) / 1000

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


def computeMetrics(ref, pred, bbox, unit):
    if not (isinstance(ref, np.ndarray) and isinstance(pred, np.ndarray) and ref.shape == pred.shape):
        print(ref.shape, pred.shape)
        return -1

    ref_choose = np.clip(ref, 0, 1)[bbox]
    pred_choose = pred[bbox]

    v1 = np.count_nonzero(ref_choose * pred_choose)
    v2 = np.count_nonzero(ref_choose) + np.count_nonzero(pred_choose)
    dice = round(2 * v1 / (v2 + 1e-8), 4)

    vd = np.sum(ref_choose - pred_choose) * unit
    rvd = (np.sum(np.abs(ref_choose - pred_choose)) / 2) / (np.sum(ref_choose) / 2 + np.sum(pred_choose) / 2)
    return dice, vd, rvd


def read3d(filePath, out_dtype=np.int16, only_header=False):
    if filePath.lower().endswith(('.nii', '.nii.gz')):
        hdr, volume = read_nii(filePath, out_dtype, only_header)
        labPath = Path(filePath)
        labPath = labPath.parent / labPath.name.replace("volume", "segmentation").replace("img", "mask")
        if not only_header:
            _, label = read_nii(labPath, out_dtype, only_header)
            label = np.clip(label, 0, 1)
        else:
            label = None
        return Image3d(volume, Header(hdr, "nii"), filePath, label)


def read_nii(file_name, out_dtype=np.int16, only_header=False):
    nib_vol = nib.load(str(file_name))
    vh = nib_vol.header
    if only_header:
        return vh, None
    affine = vh.get_best_affine()
    # assert len(np.where(affine[:3, :3].reshape(-1) != 0)[0]) == 3, affine
    trans = np.argmax(np.abs(affine[:3, :3]), axis=1)
    data = nib_vol.get_fdata().astype(out_dtype).transpose(*trans[::-1])

    if affine[0, trans[0]] > 0:  # Increase x from Right to Left
        data = np.flip(data, axis=2)
    if affine[1, trans[1]] > 0:  # Increase y from Anterior to Posterior
        data = np.flip(data, axis=1)
    if affine[2, trans[2]] < 0:  # Increase z from Interior to Superior
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


if __name__ == "__main__":
    img_file = "/home/jarvis/Downloads/000029-mask.nii.gz"
    head, img = read_nii(img_file)
    img = img[::-1].transpose(1, 0, 2)
    out_file = "/home/jarvis/Downloads/WBMRI_2018_contoured2/000029-mask_fix.nii.gz"
    write_nii(img, head, out_file)
