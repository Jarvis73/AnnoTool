import numpy as np
import pygco

from libs.common import min_window, max_window, pairwise_alpha, LABELS
from libs.model import Model3D


def graph_cut3d(data3d, seeds):
    data3d = data3d.astype(np.int16)
    seeds = seeds.astype("int8")
    unariesalt = _create_tlinks(data3d, seeds)
    pairwise = - (np.eye(2) - 1)
    pairwise = (pairwise_alpha * pairwise).astype(np.int32)

    nlinks = _create_nlinks(data3d)
    result_graph = pygco.cut_from_graph(nlinks, unariesalt.reshape(-1, 2), pairwise)
    result_labeling = result_graph.reshape(data3d.shape)
    return result_labeling


def _create_tlinks(data3d, seeds, area_weight=1, hard_constarints=True):
    tdata1, tdata2 = __similarity_for_tlinks_obj(data3d, seeds)
    if hard_constarints:
        tdata1, tdata2 = __set_hard_constraints(tdata1, tdata2, seeds)
    tdata1 = __limit(tdata1)
    tdata2 = __limit(tdata2)
    unariesalt = (0 + (np.dstack(area_weight * [tdata1.reshape(-1, 1), tdata2.reshape(-1, 1)]).copy("C"))).astype(
        np.int32)
    unariesalt = __limit(unariesalt)
    return unariesalt


def __similarity_for_tlinks_obj(data3d, seeds):
    model = Model3D()
    model.fit_from_data(data3d, seeds, [LABELS['prospect'], LABELS['background']])
    tdata1 = (-(model.linkelihood_from_data(data3d, LABELS['prospect']))) * 10  # todo
    tdata2 = (-(model.linkelihood_from_data(data3d, LABELS['background']))) * 10
    return tdata1, tdata2


def __set_hard_constraints(tdata1, tdata2, seeds):
    seeds_mask = (seeds == 1)
    tdata2[seeds_mask] = np.max(tdata2) + 1
    tdata1[seeds_mask] = 0
    seeds_mask = (seeds == 2)
    tdata1[seeds_mask] = np.max(tdata1) + 1
    tdata2[seeds_mask] = 0
    return tdata1, tdata2


def __limit(tdata):
    tdata[tdata > max_window] = max_window
    tdata[tdata < min_window] = min_window
    return tdata


def _create_nlinks(data):
    edgs_arr = __prepare_edgs_array(data)
    edges = np.vstack(edgs_arr).astype(np.int32)
    return edges


def __prepare_edgs_array(data):
    inds = np.arange(data.size).reshape(data.shape)
    edgx = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
    edgy = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
    edgz = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
    edgs_arr = [edgx, edgy, edgz]
    return edgs_arr


class GraphCut3D(object):
    pass