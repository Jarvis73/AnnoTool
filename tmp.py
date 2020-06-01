# Copyright 2019-2020 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================

import numpy as np
import nibabel as nib
from libs.graph_cut import GraphCut3D


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


def seg_gc(volume, centers=None):
    gc = GraphCut3D(volume, [1, 1, 1])
    # seeds = np.zeros_like(volume, np.uint8)
    # fg = np.array(centers["fg"], np.int32).reshape(-1, 3)
    # bg = np.array(centers["bg"], np.int32).reshape(-1, 3)
    # if fg.shape[0] > 0:
    #     seeds[fg[:, 0], fg[:, 1], fg[:, 2]] = 1
    # if bg.shape[0] > 0:
    #     seeds[bg[:, 0], bg[:, 1], bg[:, 2]] = 2
    seeds = np.load("seeds.npy")
    gc.set_seeds(seeds)
    gc.run()
    return gc.segmentation


vol = read_nii("009.nii")[1]
seg_gc(vol)
