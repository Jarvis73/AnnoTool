import cv2
import grpc
import time
import numpy as np
import nibabel as nib
import qimage2ndarray as q2a
import skimage.measure as measure
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import geodesic_distance as gd
import matplotlib.pyplot as plt


class Image3d(object):
	""" For 3D gray image """

	BK_DEFAULT, BK_FIRST, BK_SECOND = -1, -2, -3

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
		self._alpha = 0.5
		self._color = np.array([0, 255, 0])
		self.contour = False
		self.backup = None
		self.backup_idx = Image3d.BK_DEFAULT
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
		dirty = False
		if low is not None and isinstance(low, (int, float)) and self.low != low:
			self.low = low
			dirty = True
		if high is not None and isinstance(high, (int, float)) and self.high != high:
			self.high = high
			dirty = True

	def at(self, i, axis=0, show=True, showLabel2=True):
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
				if showLabel2:
					obj = np.clip(self.segCache[i], 0, 1)
				else:
					obj = np.clip(self.segCache[i] - 1, 0, 1)
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

	def gunet(self, i, centers=None, stddevs=None, name=None):
		if i == 0:
			images = np.concatenate([np.zeros(self.shape[1:] + (1,)), self.volume[:2].transpose(1, 2, 0)], axis=-1)
		elif i == self.shape[self.axis] - 1:
			images = np.concatenate([self.volume[-2:].transpose(1, 2, 0), np.zeros(self.shape[1:] + (1,))], axis=-1)
		else:
			images = self.volume[i - 1:i + 2].transpose(1, 2, 0)
		if name == "012_gnet_sp_f0":
			output = gunet_liver_segmentation(images, centers, stddevs, 512, 512, name)
		else:
			output = gunet_segmentation(images, centers, stddevs, 960, 320)
		if output is None:
			return None, None
		print(self.filePath, i, time)

		if i in self.segCache:
			self.backup = self.segCache[i]
			self.backup_idx = i
			self.segCache[i] = np.maximum(self.segCache[i], output)
		else:
			self.segCache[i] = output
			self.backup_idx = i
			self.backup = None

		return True, time


	def gnnu2d(self, name, i, centers=None, stddevs=None):
		output, time = gnnu2d_segmentation(self.volume, i, centers, stddevs, 20, 1024, 320, self.guide_temp, name)
		if output is None:
			return None

		if i in self.segCache:
			self.backup = self.segCache[i]
			self.backup_idx = i
			self.segCache[i] = np.maximum(self.segCache[i], output)
		else:
			self.segCache[i] = output
			self.backup_idx = i
			self.backup = None

		return True

	def gnnu3d(self, name, centers=None, stddevs=None):
		output = gnnu3d_segmentation(self.volume, centers, stddevs, 20, 512, 160, self.guide_temp, name)
		if output is None:
			return None

		if len(self.segCache) == 0:
			self.segCache = output
		else:
			self.backup = self.segCache
			self.segCache = np.maximum(self.segCache, output)

		return True

	def undo(self):
		if self.backup_idx >= 0:
			if self.backup is not None:
				self.segCache[self.backup_idx] = self.backup
				self.backup = None
				self.backup_idx = Image3d.BK_DEFAULT
			else:
				self.segCache.pop(self.backup_idx)
		elif self.backup_idx == Image3d.BK_FIRST:
			self.segCache = {}
			self.backup_idx = Image3d.BK_DEFAULT
		elif self.backup_idx == Image3d.BK_SECOND:
			self.segCache = self.backup
			self.backup = None
			self.backup_idx = Image3d.BK_DEFAULT
		return self.backup_idx


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


def create_gaussian_distribution_v2(shape, centers, stddevs, indexing="ij", keepdims=False, geo=None):
    """
    Parameters
    ----------
    shape: list
        two values
    centers: ndarray
        Float ndarray with shape [n, d], d means (x, y, ...)
    stddevs: ndarray
        Float ndarray with shape [n, d], d means (x, y, ...)
    indexing: {'xy', 'ij'}, optional
        Cartesian ('xy') or matrix ('ij', default) indexing of output.
        See Notes for more details.
    keepdims: bool
        Keep final dimension

    TODO(zjw) Warning: indexing='xy' need test

    Returns
    -------
    A batch of spatial guide image with shape [h, w, 1] for 2D and [d, h, w, 1] for 3D

    Notes
    -----
    -1s in center and stddev are padding value and almost don't affect spatial guide
    """
    centers = np.asarray(centers, np.float32)                                   # [n, 2]    if dimension=2
    stddevs = np.asarray(stddevs, np.float32)                                   # [n, 2]
    assert centers.ndim == 2, centers.shape
    assert centers.shape == stddevs.shape, (centers.shape, stddevs.shape)
    coords = [np.arange(0, s) for s in shape]
    coords = np.tile(
        np.stack(np.meshgrid(*coords, indexing=indexing), axis=-1)[None],
        reps=[centers.shape[0]] + [1] * (centers.shape[1] + 1))                 # [n, h, w, 2]
    coords = coords.astype(np.float32)

    if geo is None:
        if len(shape) == 2:
            centers = centers[..., None, None, :]                                       # [n, 1, 1, 2]
            stddevs = stddevs[..., None, None, :]                                       # [n, 1, 1, 2]
        elif len(shape) == 3:
            centers = centers[..., None, None, None, :]  # [n, 1, 1, 2]
            stddevs = stddevs[..., None, None, None, :]  # [n, 1, 1, 2]
        else:
            raise ValueError()
        normalizer = 2 * stddevs * stddevs                                          # [n, 1, 1, 2]
        d = np.exp(-np.sum((coords - centers) **2 / normalizer, axis=-1, keepdims=keepdims))   # [n, h, w, 1] / [n, h, w]
        guide = np.max(d, axis=0)
    else:
        S = np.zeros(shape, dtype=np.uint8)
        for c in centers:
            S[int(c[0]), int(c[1])] = 1
        geo_dis = gd.geodesic2d_raster_scan(geo, S, 1.0, 2)
        guide = np.exp(-geo_dis ** 2)
    # plt.imshow(guide)
    # plt.show()
    return guide


def gunet_segmentation(triple, centers, stddevs, height, width):
	h, w, d = triple.shape
	img_input = triple.astype(np.float32)
	img_input = np.clip(img_input, 0, 900) / 900

	if centers is None or stddevs is None or len(centers) == 0 or len(stddevs) == 0:
		guide = np.ones((h, w), dtype=np.float32) * 0.5
	else:
		if isinstance(centers, dict):
			guide_fg = create_gaussian_distribution_v2((h, w), centers["fg"], stddevs["fg"]) \
				if len(centers["fg"]) > 0 else 0
			guide_bg = create_gaussian_distribution_v2((h, w), centers["bg"], stddevs["bg"]) \
				if len(centers["bg"]) > 0 else 0
			guide = 0.5 + (guide_fg - guide_bg) * 0.425
		else:
			guide = create_gaussian_distribution_v2((h, w), centers, stddevs)
	inputs = np.concatenate((img_input, guide[..., None]), axis=-1)
	inputs = cv2.resize(inputs, (width, height), interpolation=cv2.INTER_LINEAR)

	host = "localhost"
	port = 8500
	channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

	request = predict_pb2.PredictRequest()
	request.model_spec.name = "112_nf_sp_dp"
	request.model_spec.signature_name = "pred_images"
	request.inputs['height'].CopyFrom(tf.make_tensor_proto(height))
	request.inputs['width'].CopyFrom(tf.make_tensor_proto(width))
	request.inputs['images_bytes'].CopyFrom(tf.make_tensor_proto(inputs[..., :3], shape=[1, height, width, 3]))
	request.inputs['guide_bytes'].CopyFrom(tf.make_tensor_proto(inputs[..., 3:], shape=[1, height, width, 1]))

	try:
		result = stub.Predict(request)
	except:
		result = None

	# Reference:
	# How to access nested values
	# https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
	# logits = np.array(result.outputs["output_bytes2"].float_val).reshape(height, width, 2)
	if result is not None:
		output = np.array(result.outputs["output_bytes"].int64_val).reshape(height, width)
		output = cv2.resize(output, (w, h), interpolation=cv2.INTER_NEAREST)
		return output
	else:
		return None


def gunet_liver_segmentation(triple, centers, stddevs, height, width, name):
	h, w, d = triple.shape
	img_input = triple.astype(np.float32)
	img_input = (np.clip(img_input, -200, 250) + 200) / 450

	if centers is None or stddevs is None or len(centers) == 0 or len(stddevs) == 0:
		guide = np.ones((h, w), dtype=np.float32) * 0
	else:
		if isinstance(centers, dict):
			guide_fg = create_gaussian_distribution_v2((h, w), centers["fg"], stddevs["fg"]) \
				if len(centers["fg"]) > 0 else 0
			guide_bg = create_gaussian_distribution_v2((h, w), centers["bg"], stddevs["bg"]) \
				if len(centers["bg"]) > 0 else 0
			guide = guide_fg - guide_bg
		else:
			guide = create_gaussian_distribution_v2((h, w), centers, stddevs) * 0.85
	inputs = np.concatenate((img_input, guide[..., None]), axis=-1)
	if height != inputs.shape[0] or width != inputs.shape[1]:
		inputs = cv2.resize(inputs, (width, height), interpolation=cv2.INTER_LINEAR)

	host = "localhost"
	port = 8500
	channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

	request = predict_pb2.PredictRequest()
	request.model_spec.name = name
	request.model_spec.signature_name = "serving_default"
	request.inputs['height'].CopyFrom(tf.make_tensor_proto(height))
	request.inputs['width'].CopyFrom(tf.make_tensor_proto(width))
	request.inputs['images_bytes'].CopyFrom(tf.make_tensor_proto(inputs[..., :3], shape=[1, height, width, 3]))
	request.inputs['guide_bytes'].CopyFrom(tf.make_tensor_proto(inputs[..., 3:], shape=[1, height, width, 1]))

	try:
		result = stub.Predict(request)
	except:
		result = None

	# Reference:
	# How to access nested values
	# https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
	# logits = np.array(result.outputs["output_bytes2"].float_val).reshape(height, width, 2)
	if result is not None:
		output = np.array(result.outputs["output_bytes"].int64_val).reshape(height, width)
		output = cv2.resize(output, (w, h), interpolation=cv2.INTER_NEAREST)
		return output
	else:
		return None


def gnnu3d_segmentation(image, centers, stddevs, depth, height, width, guide_temp, name):
    d, h, w = image.shape
    img_input = image.astype(np.float32)
    #preprocessing...
    img_input = (img_input - img_input.mean()) / (img_input.std() + 1e-8)

    if centers is None or stddevs is None or len(centers) == 0 or len(stddevs) == 0:
        guide = np.zeros((d, h, w), dtype=np.float32)
    else:
        guide = create_gaussian_distribution_v2((d, h, w), centers, stddevs)
    inputs = np.concatenate((img_input[None], guide[None]), axis=0)
    inputs = inputs.reshape((-1, h, w))
    inputs = inputs.transpose((1, 2, 0))
    inputs = cv2.resize(inputs, (width, height), interpolation=cv2.INTER_LINEAR)
    inputs = inputs.transpose((2, 0, 1))
    inputs = inputs.reshape(2, depth, height, width)

    host = "localhost"
    port = 8500
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = name
    request.model_spec.signature_name = "serving_default"
    request.inputs['in'].CopyFrom(tf.make_tensor_proto(inputs[None], shape=[1, 2, depth, height, width]))

    try:
        result = stub.Predict(request)
    except:
        result = None

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    # logits = np.array(result.outputs["output_bytes2"].float_val).reshape(height, width, 2)
    if result is not None:
        output = np.array(result.outputs["out"].int64_val).reshape(depth, height, width)
        output = output.transpose((1, 2, 0))
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_NEAREST)
        output = output.transpose((2, 0, 1))
        return output
    else:
        return None


def gnnu2d_segmentation(image, idx, centers, stddevs, depth, height, width, guide_temp, name):
    d, h, w = image.shape
    img_input = image[idx:idx + 1].astype(np.float32)
    #preprocessing...
    img_input = (img_input - img_input.mean()) / (img_input.std() + 1e-8)

    if centers is None or stddevs is None or len(centers) == 0 or len(stddevs) == 0:
	    guide = np.ones((h, w), dtype=np.float32)
    else:
	    guide = create_gaussian_distribution_v2((h, w), centers, stddevs) * 0.85
    inputs = np.concatenate((img_input, guide[None]), axis=0).transpose((1, 2, 0))
    inputs = cv2.resize(inputs, (width, height), interpolation=cv2.INTER_LINEAR).transpose((2, 0, 1))

    host = "localhost"
    port = 8500
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = name
    request.model_spec.signature_name = "serving_default"
    request.inputs['in'].CopyFrom(tf.make_tensor_proto(inputs[None], shape=[1, 2, height, width]))

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
	    output = np.array(result.outputs["out"].int64_val).reshape(height, width)
	    output = cv2.resize(output, (w, h), interpolation=cv2.INTER_NEAREST)
	    return output
    else:
	    return None


def createEllipse(rect, shape):
	x, y, w, h = rect
	cy, cx = h // 2, w // 2
	xrng = np.arange(w)
	yrng = np.arange(h)
	coord = np.stack(np.meshgrid(yrng, xrng, indexing="ij"), axis=-1)
	ellipse_mask = np.zeros(shape, dtype=np.uint8)
	ellipse_mask[y:y + h, x:x + w] = np.sum(((coord - [cy, cx]) / [cy, cx]) ** 2, axis=-1) <= 1
	return ellipse_mask


def compute_robust_moments(rect, shape, isotropic=False, indexing="ij", min_std=0.):
    """
    Compute robust center and standard deviation of a binary image(0: background, 1: foreground).

    Support n-dimension array.

    Parameters
    ----------
    binary_image: ndarray
        Input a image
    isotropic: boolean
        Compute isotropic standard deviation or not.
    indexing: {'xy', 'ij'}, optional
        Cartesian ('xy') or matrix ('ij', default) indexing of output.
        See Notes for more details.
    min_std: float
        Set stddev lower bound

    Returns
    -------
    center: ndarray
        A vector with dimension = `binary_image.ndim`. Median of all the points assigned 1.
    std_dev: ndarray
        A vector with dimension = `binary_image.ndim`. Standard deviation of all the points assigned 1.

    Notes
    -----
    All the points assigned 1 are considered as a single object.

    """
    binary_image = createEllipse(rect, shape)
    ndim = binary_image.ndim
    coords = np.nonzero(binary_image)
    points = np.asarray(coords).astype(np.float32)
    if points.shape[1] == 0:
        return np.array([-1.0] * ndim, dtype=np.float32), np.array([-1.0] * ndim, dtype=np.float32)
    points = np.transpose(points)       # [pts, 2], 2: (i, j)
    center = np.median(points, axis=0)  # [2]

    # Compute median absolute deviation(short for mad) to estimate standard deviation
    if isotropic:
        diff = np.linalg.norm(points - center, axis=1)
        mad = np.median(diff)
        mad = np.array([mad] * ndim)
    else:
        diff = np.absolute(points - center)
        mad = np.median(diff, axis=0)
    std_dev = 1.4826 * mad
    std_dev = np.maximum(std_dev, [min_std] * ndim)
    if not indexing or indexing == "xy":
        return center[::-1], std_dev[::-1]
    elif indexing == "ij":
        return center, std_dev
    else:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")


if __name__ == "__main__":
	rect = 250, 250, 50, 100
	shape = 512, 512
	mask0 = createEllipse(rect, shape)
	center, stddev = compute_robust_moments(rect, shape)
	mask = create_gaussian_distribution_v2(shape, [center], [stddev])
	plt.subplot(121)
	plt.imshow(mask0)
	plt.subplot(122)
	plt.imshow(mask)
	plt.show()
