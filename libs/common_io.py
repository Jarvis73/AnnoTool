#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from libs.constants import DEFAULT_ENCODING
from libs.ustr import ustr
from libs.shape import Shape


COMM_EXT = '.cxml'  # common xml
ENCODE_METHOD = DEFAULT_ENCODING

"""
    CommonWriter/CommonReader removes 'difficult' property and adds 
    more types of interactions:
        - rectangle
        - point
"""


class CommonWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.elllist = []
        self.pntlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, **kwargs):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox.update(kwargs)
        self.boxlist.append(bndbox)

    def addPnt(self, x, y, name, **kwargs):
        pnt = {'x': x, 'y': y}
        pnt['name'] = name
        pnt.update(kwargs)
        self.pntlist.append(pnt)

    def savebox(self, top, shapeList, cls):
        for each_object in shapeList:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = ustr(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin']))== 1):
                truncated.text = "1" # max == height or min
            elif (int(float(each_object['xmax']))==int(float(self.imgSize[1]))) or (int(float(each_object['xmin']))== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            bndbox = SubElement(object_item, cls)
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])
            zmin = SubElement(bndbox, "zmin")
            zmin.text = str(each_object["z1"])
            zmax = SubElement(bndbox, "zmax")
            zmax.text = str(each_object["z2"])
            if "axis" in each_object:
                axis = SubElement(bndbox, "axis")
                axis.text = str(each_object["axis"])
            if "slice" in each_object:
                slice = SubElement(bndbox, "slice")
                slice.text = str(each_object["slice"])

    def appendObjects(self, top):
        self.savebox(top, self.boxlist, 'bndbox')

        for each_object in self.pntlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = ustr(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            truncated.text = "0"
            pnt = SubElement(object_item, 'pnt')
            x = SubElement(pnt, 'x')
            x.text = str(each_object['x'])
            y = SubElement(pnt, 'y')
            y.text = str(each_object['y'])
            fg = SubElement(pnt, 'fg')
            fg.text = str(each_object['fg'])
            if "axis" in each_object:
                axis = SubElement(pnt, "axis")
                axis.text = str(each_object["axis"])
            if "slice" in each_object:
                slice = SubElement(pnt, "slice")
                slice.text = str(each_object["slice"])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + COMM_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class CommonReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseCXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addBndBox(self, label, bndbox, dtype):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        zmin = int(float(bndbox.find('zmin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        zmax = int(float(bndbox.find('zmax').text))
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        shape = {
            "type": dtype,
            "label": label,
            "shape": points,
            "color1": None,
            "color2": None,
            "z1": zmin,
            "z2": zmax
        }

        axis = bndbox.find("axis")
        if axis is not None:
            shape["axis"] = int(axis.text)

        slice_ = bndbox.find("slice")
        if slice_ is not None:
            shape["slice"] = int(slice_.text)

        self.shapes.append(shape)

    def addPnt(self, label, pnt):
        x = int(float(pnt.find('x').text))
        y = int(float(pnt.find('y').text))
        points = [(x, y)]

        shape = {
            "type": Shape.POINT,
            "label": label,
            "shape": points,
            "color1": None,
            "color2": None,
            "fg": pnt.find('fg').text == "True"
        }

        axis = pnt.find("axis")
        if axis is not None:
            shape["axis"] = int(axis.text)

        slice_ = pnt.find("slice")
        if slice_ is not None:
            shape["slice"] = int(slice_.text)

        self.shapes.append(shape)

    def parseCXML(self):
        assert self.filepath.endswith(COMM_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            label = object_iter.find('name').text

            bndbox = object_iter.find("bndbox")
            if bndbox is not None:
                self.addBndBox(label, bndbox, Shape.RECTANGLE)
                continue
            
            pnt = object_iter.find("pnt")
            if pnt is not None:
                self.addPnt(label, pnt)
                continue

        return True
