# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

from PyQt5.QtGui import QImage
from base64 import b64encode, b64decode
from libs.common_io import CommonWriter
from libs.shape import Shape
from libs.image3d import read3d
import os.path
import sys


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = ".cxml"

    def __init__(self):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.verified = False

    def saveCommonFormat(self, filename, shapes, imagePath):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        # Read from file path because self.imageData might be empty if saving to
        is3d = imagePath.lower().endswith(tuple([".nii", ".nii.gz"]))
        if is3d:
            z, y, x = read3d(imagePath, only_header=True).shape
            imageShape = y, x, z
        else:
            image = QImage()
            image.load(imagePath)
            imageShape = [image.height(), image.width(),
                          1 if image.isGrayscale() else 3]
        writer = CommonWriter(imgFolderName, imgFileName,
                              imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            dtype = shape['dtype']
            ext_items = {}
            if is3d:
                ext_items["axis"] = shape["axis"]
                ext_items["slice"] = shape["slice"]
                ext_items["z1"] = shape["z1"]
                ext_items["z2"] = shape["z2"]
                if "fg" in shape:
                    ext_items["fg"] = shape["fg"]
            if dtype == Shape.RECTANGLE:
                bndbox = LabelFile.convertPoints2BndBox(points)
                writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, **ext_items)
            elif dtype == Shape.POINT:
                pnt = points[0]
                writer.addPnt(int(pnt[0]), int(pnt[1]), label, **ext_items)
        writer.save(targetFile=filename)
        return

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return int(xmin), int(ymin), int(xmax), int(ymax)
