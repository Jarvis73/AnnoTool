import numpy as np
from PyQt5.QtCore import Qt

QString = str

gmm_cvtype = "covariance_type"
gmm_cvtype_bad = "cvtype"
default_model_params = {'type': 'gmm_same', 'params': {}, 'fv_type': 'intensity'}

# qt_utils
VIEW_TABLE = {"axial": (2, 1, 0), "sagittal": (1, 0, 2), "coronal": (2, 0, 1)}

BOX_BUTTONS_SEED_SHIFT_OFFSET = 2

# BGRA order
GRAY_COLORTABLE = np.array([[ii, ii, ii, 255] for ii in range(256)], dtype=np.uint8)
SEEDS_COLORTABLE = np.array([[0, 255, 0, 220], [64, 0, 255, 220], [0, 200, 128, 220], [64, 128, 200, 220]],
                            dtype=np.uint8)

CONTOURS_COLORS = {1: [64, 255, 0], 2: [255, 0, 64], 3: [0, 64, 255], 4: [255, 64, 0], 5: [64, 0, 255], 6: [0, 255, 64],
                   7: [0, 128, 192], 8: [128, 0, 192], 9: [128, 192, 0], 10: [0, 192, 128], 11: [192, 128, 0],
                   12: [192, 0, 128], 13: [128, 0, 0]}
CONTOURS_COLORTABLE = np.zeros((256, 4), dtype=np.uint8)
CONTOURS_COLORTABLE[:, :3] = 255
CONTOURLINES_COLORTABLE = np.zeros((256, 2, 4), dtype=np.uint8)
CONTOURLINES_COLORTABLE[:, :, :3] = 255
for ii, jj in CONTOURS_COLORS.items():
    key = ii - 1
    CONTOURS_COLORTABLE[key, :3] = jj
    CONTOURS_COLORTABLE[key, 3] = 64
    CONTOURLINES_COLORTABLE[key, 0, :3] = jj
    CONTOURLINES_COLORTABLE[key, 0, 3] = 16
    CONTOURLINES_COLORTABLE[key, 1, :3] = jj
    CONTOURLINES_COLORTABLE[key, 1, 3] = 255

DRAW_MASK = [(np.array([[1]], dtype=np.int8), "small pen"),
             (np.array([[0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0]], dtype=np.int8), "middle pen"),
             (np.array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int8), "large pen",)]

BOX_BUTTONS_DRAW = {Qt.LeftButton: 1, Qt.RightButton: 0}

BOX_BUTTONS_SEED = {Qt.LeftButton: 1, Qt.RightButton: 2}
