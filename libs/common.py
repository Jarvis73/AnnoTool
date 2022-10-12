import numpy as np
from PyQt5.QtCore import Qt

LABELS = {'all eraser': 0, 'prospect': 1, 'background': 2}

min_window = 0
max_window = 1000

# segmentation params
pairwise_alpha = 20

# model
fv_type = 'intensity'
model_type = 'gmm_same'

# view
VIEW_TABLE = {"axial": (2, 1, 0), "sagittal": (1, 0, 2), "coronal": (2, 0, 1)}
SLAB = {'0': 0, '1': 1}
MAX_HEIGHT = 600
DRAW_MASK = {"small pen": np.array([[1]], dtype=np.int8)}
COMBE_CONTOURS_OPTIOM = ["fill", "contours", "off"]

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

BOX_BUTTONS_SEED_SHIFT_OFFSET = 2
BOX_BUTTONS_SEED = {Qt.LeftButton: 1, Qt.RightButton: 2}
