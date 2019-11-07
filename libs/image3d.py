import numpy as np
import nibabel as nib
import qimage2ndarray as q2a


class Image3d(object):
	""" For 3D gray image """

	def __init__(self, volume, meta=None):
		# self.raw = raw
		# self.volume = raw.copy()
		if volume is None and meta is None:
			raise ValueError("Both `volume` and `meta` are None.")
		self.volume = volume
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

	def __getitem__(self, i):
		return self.volume[i]

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

	def at(self, i, axis=0):
		# We don't check index out of bounds
		if axis == 0:
			img = self.volume[i]
		elif axis == 1:
			img = self.volume[:, i]
		elif axis == 2:
			img = self.volume[:, :, i]
		return q2a.gray2qimage(img, normalize=(self.low, self.high))

	def pixel(self, slice_idx, i, j, axis=0):
		if axis == 0:
			return self.volume[slice_idx, i, j]
		elif axis == 1:
			return self.volume[i, slice_idx, j]
		elif axis == 2:
			return self.volume[i, j, slice_idx]
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


def read3d(filePath, out_dtype=np.int16, only_header=False):
	if filePath.lower().endswith(('.nii', '.nii.gz')):
		hdr, volume = read_nii(filePath, out_dtype, only_header)
		return Image3d(volume, Header(hdr, "nii"))


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
