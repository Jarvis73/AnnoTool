#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import distutils.spawn
import os.path
import platform
import re
import sys
import subprocess
import time

from functools import partial
from collections import defaultdict
import numpy as np
import cv2

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import *
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.toolBar import ToolBar
from libs.common_io import CommonReader, COMM_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem, HashableQTableWidgetItem
from libs.image3d import read3d, Image3d, DSKey, write_nii, read_nii, computeMetrics


__appname__ = 'labelImg'
TWO_D, THREE_D = list(range(2))


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)

        # Save as Pascal voc xml
        self.defaultSaveDir = defaultSaveDir
        # Save segmentation
        self.defaultSegDir = None

        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # For image type, natural image(png, bmp, pbm, pgm, ppm, xbm, xpm) & 
        # medical image(nii) are supported
        # 
        # Other natural image formats need extra plugins
        # QtCore.QCoreApplication.addLibraryPath(r"<CondaEnvPath>\Library\plugins")
        self.dim = TWO_D
        self.i3d: Image3d = None     # image 3d instance
        self.idx = 0
        self.axis = 0
        self.total_time = 0

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Widgets and related state for naming labels.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.shapesToOthers = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        #################################################################################
        # For 3d image: settings
        i3dSettingLayout = QVBoxLayout()
        i3dSettingLayout.setContentsMargins(0, 0, 0, 0)

        # window/level
        self.int_low_str = QLabel(getStr("intLow"))
        self.int_high_str = QLabel(getStr("intHigh"))
        self.int_low = QSpinBox()
        self.int_high = QSpinBox()
        self.int_low.setRange(-10000, 10000)
        self.int_high.setRange(-10000, 10000)
        self.int_low.setValue(settings.get(SETTING_INT_MIN, 0))
        self.int_high.setValue(settings.get(SETTING_INT_MAX, 600))
        self.int_low.valueChanged.connect(self.update3dImageLow)
        self.int_high.valueChanged.connect(self.update3dImageHigh)
        lowHighQHBoxLayout = QHBoxLayout()
        lowHighQHBoxLayout.addWidget(self.int_low_str)
        lowHighQHBoxLayout.addWidget(self.int_low)
        lowHighQHBoxLayout.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        lowHighQHBoxLayout.addWidget(self.int_high_str)
        lowHighQHBoxLayout.addWidget(self.int_high)
        self.lowHighContainer = QWidget()
        self.lowHighContainer.setLayout(lowHighQHBoxLayout)
        i3dSettingLayout.addWidget(self.lowHighContainer)

        self.segGTColor = QPushButton("GT Color")
        self.segGTColor.setStyleSheet("background-color: #FF0000; color: white")
        self.segGTColor.clicked.connect(self.segResColorPickerGT)
        self.segResColor = QPushButton("Seg Color")
        self.segResColor.setStyleSheet("background-color: #FFFF00; color: black")
        self.segResColor.clicked.connect(self.segResColorPicker)
        self.segResMode = QPushButton("Mask")
        self.segResMode.clicked.connect(self.segResModeChanged)
        segResLayout = QHBoxLayout()
        segResLayout.addWidget(self.segGTColor)
        segResLayout.addWidget(self.segResColor)
        segResLayout.addWidget(self.segResMode)
        self.segResContainer = QWidget()
        self.segResContainer.setLayout(segResLayout)
        i3dSettingLayout.addWidget(self.segResContainer)

        self.segResSliderLabel = QLabel("Alpha: ")
        self.segResSlider = QSlider(Qt.Horizontal)
        self.segResSlider.setValue(settings.get(SETTING_ALPHA, 0.5))
        self.segResSlider.valueChanged.connect(self.segResSliderChanged)
        self.segAutoCheckBox = QCheckBox("Auto: ")
        self.segAutoCheckBox.setChecked(settings.get(SETTING_AUTO_SEG, False))
        self.segAutoCheckBox.setLayoutDirection(Qt.RightToLeft)
        self.labelShowCheckBox = QCheckBox("Show Label:")
        self.labelShowCheckBox.setChecked(True)
        self.labelShowCheckBox.setLayoutDirection(Qt.RightToLeft)
        self.labelShowCheckBox.stateChanged.connect(self.labelShowStateChanged)
        segResLayout2 = QHBoxLayout()
        segResLayout2.addWidget(self.segResSliderLabel)
        segResLayout2.addWidget(self.segResSlider)
        segResLayout2.addWidget(self.segAutoCheckBox)
        segResLayout2.addWidget(self.labelShowCheckBox)
        self.segResContainer2 = QWidget()
        self.segResContainer2.setLayout(segResLayout2)
        i3dSettingLayout.addWidget(self.segResContainer2)

        # segmentation algorithmDann
        segAlgLabel = QLabel(getStr("segAlgLabel"))
        self.segAlg = ["DIN", "EDT", "GDT", "Graph Cut 3D", "Random Walk 3D"]
        self.segName = ["din", "euc", "geo", "GraphCut3D", "RandomWalk3D"]
        self.segAlgComboBox = QComboBox()
        self.segAlgComboBox.addItems(self.segAlg)
        self.gtShowCheckBox = QCheckBox("GT:")
        self.gtShowCheckBox.setChecked(False)
        self.gtShowCheckBox.setLayoutDirection(Qt.RightToLeft)
        self.gtShowCheckBox.setShortcut(QKeySequence(Qt.Key_M))
        self.gtShowCheckBox.stateChanged.connect(self.updateCanvasImage)
        self.segShowCheckBox = QCheckBox("Seg:")
        self.segShowCheckBox.setChecked(True)
        self.segShowCheckBox.setLayoutDirection(Qt.RightToLeft)
        self.segShowCheckBox.setShortcut(QKeySequence(Qt.Key_S))
        self.segShowCheckBox.stateChanged.connect(self.updateCanvasImage)
        self.segComputeDiceButton = QPushButton("Dice")
        self.segComputeDiceButton.setShortcut(QKeySequence(Qt.Key_Space))
        self.segComputeDiceButton.clicked.connect(self.computeDiceClicked)
        self.stddev_str = QLabel("stddev:")
        self.stddev = QSpinBox()
        self.stddev.setRange(1, 50)
        self.stddev.setValue(settings.get(SETTING_STDDEV, 7))

        segAlgLayout = QHBoxLayout()
        segAlgLayout.addWidget(segAlgLabel)
        segAlgLayout.addWidget(self.segAlgComboBox)
        segAlgLayout.addWidget(self.gtShowCheckBox)
        segAlgLayout.addWidget(self.segShowCheckBox)
        segAlgLayout.addWidget(self.segComputeDiceButton)
        segAlgLayout.addWidget(self.stddev_str)
        segAlgLayout.addWidget(self.stddev)
        segAlgLayout.addItem(QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.segAlgContainer = QWidget()
        self.segAlgContainer.setLayout(segAlgLayout)
        i3dSettingLayout.addWidget(self.segAlgContainer)

        self.segAlgButtonRun = QPushButton(getStr("segAlgButtonRun"))
        self.segAlgButtonRun.clicked.connect(self.segRun)
        self.segAlgButtonRun.setShortcut(QKeySequence(Qt.Key_R))
        self.segAlgButtonRun.setEnabled(False)
        self.segAlgButtonClear = QPushButton(getStr("segAlgButtonClear"))
        self.segAlgButtonClear.clicked.connect(self.segClear)
        self.segAlgButtonClear.setEnabled(False)
        self.segAlgButtonUndo = QPushButton(getStr("segAlgButtonUndo"))
        self.segAlgButtonUndo.clicked.connect(self.segUndo)
        self.segAlgButtonUndo.setShortcut(QKeySequence(Qt.Key_U))
        self.segAlgButtonUndo.setEnabled(False)
        self.segAlgButtonSave = QPushButton(getStr("segAlgButtonSave"))
        self.segAlgButtonSave.clicked.connect(self.segSave)
        self.segAlgButtonSave.setEnabled(False)
        segAlgLayout2 = QHBoxLayout()
        segAlgLayout2.addWidget(self.segAlgButtonRun)
        segAlgLayout2.addWidget(self.segAlgButtonClear)
        segAlgLayout2.addWidget(self.segAlgButtonUndo)
        segAlgLayout2.addWidget(self.segAlgButtonSave)
        self.segAlgContainer2 = QWidget()
        self.segAlgContainer2.setLayout(segAlgLayout2)
        i3dSettingLayout.addWidget(self.segAlgContainer2)

        i3dSettingContainer = QGroupBox()
        i3dSettingContainer.setStyleSheet("QGroupBox{padding-top:15px; margin-top:-15px}")
        # i3dSettingContainer = QWidget()
        i3dSettingContainer.setLayout(i3dSettingLayout)

        self.settingDock = QDockWidget(getStr('i3dSetting'), self)
        self.settingDock.setObjectName("3dSettingObj")
        self.settingDock.setWidget(i3dSettingContainer)
        self.settingDock.setFeatures(QDockWidget.DockWidgetFloatable)
        self.settingDock.setVisible(False)

        #################################################################################

        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(getStr('useDefaultLabel'))
        self.useDefaultLabelCheckbox.setChecked(True)
        self.defaultLabelTextLine = QLineEdit()
        if len(self.labelHist) > 0:
            self.defaultLabelTextLine.setText(self.labelHist[0])
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit button
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(useDefaultLabelContainer)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        # For multiple columns
        table = QTableWidget()
        table.itemActivated.connect(self.labelSelectionChanged)
        table.itemSelectionChanged.connect(self.labelSelectionChanged)
        # table.itemDoubleClicked.connect(self.editLabel)
        table.itemChanged.connect(self.labelItemChanged)
        font = QFont()
        font.setBold(True)
        table.horizontalHeader().setFont(font)
        # table.setFrameShape(QFrame.NoFrame)
        table.horizontalHeader().setFixedHeight(25)
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setColumnCount(5)
        table.horizontalHeader().resizeSection(0, 40)
        table.horizontalHeader().resizeSection(1, 80)
        table.horizontalHeader().resizeSection(2, 80)
        table.horizontalHeader().resizeSection(3, 80)
        table.setHorizontalHeaderLabels([getStr('tableHeaderAxis'), getStr('tableHeaderSlice'),
                                         getStr('tableHeaderZ1'), getStr('tableHeaderZ2'),
                                         getStr('tableHeaderLabel')])
        table.horizontalHeader().setSectionsClickable(True)
        # table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.labelTable = table
        listLayout.addWidget(self.labelTable)

        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newRect.connect(self.newRect)
        self.canvas.newPoint.connect(self.newPoint)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.settingDock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        self.esc_func = QShortcut(QKeySequence('Esc'), self)
        self.esc_func.activated.connect(self.setEdit)

        # Actions
        action = partial(newAction, self)
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        open = action(getStr('openFile'), self.openFile,
                      'Ctrl+O', 'open', getStr('openFileDetail'))

        opendir = action(getStr('openDir'), self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        changeSavedir = action(getStr('changeSaveDir'), self.changeSavedirDialog,
                               'Ctrl+r', 'open', getStr('changeSavedAnnotationDir'))

        openAnnotation = action(getStr('openAnnotation'), self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', getStr('openAnnotationDetail'))

        openNextImg = action(getStr('nextImg'), self.openNextImg,
                             'd', 'next', getStr('nextImgDetail'))

        openPrevImg = action(getStr('prevImg'), self.openPrevImg,
                             'a', 'prev', getStr('prevImgDetail'))

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+S', 'save', getStr('saveDetail'), enabled=False)

        saveAs = action(getStr('saveAs'), self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', getStr('saveAsDetail'), enabled=False)

        close = action(getStr('closeCur'), self.closeFile, 'Ctrl+W', 'close', getStr('closeCurDetail'))

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createRect = action(getStr('crtBox'), self.createRect,
                        'w', 'new', getStr('crtBoxDetail'), enabled=False)
        createPoint = action(getStr('crtPoint'), self.createPoint,
                             'f', 'new', getStr('crtPointDetail'), enabled=False)
        createPointMode = action(getStr('crtPoint'), self.setCreatePointMode,
                                 'f', 'new', getStr('crtPointDetail'), enabled=False)
        delete = action(getStr('delBox'), self.deleteSelectedShape,
                        'Delete', 'delete', getStr('delBoxDetail'), enabled=False)
        copy = action(getStr('dupBox'), self.copySelectedShape,
                      'Ctrl+D', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        advancedMode = action(getStr('advancedMode'), self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', getStr('advancedModeDetail'),
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel,
                      'Ctrl+E', 'edit', getStr('editLabelDetail'),
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, saveAs=saveAs, open=open, close=close, resetAll=resetAll,
                              lineColor=color1, createRect=createRect, createPoint=createPoint,
                              delete=delete, edit=edit, copy=copy,
                              createPointMode=createPointMode,
                              advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, close, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.drawSquaresOption),
                              beginnerContext=(createRect, createPoint, edit, copy, delete),
                              advancedContext=(createRect, createPointMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, createRect, createPoint,
                                  createRect, createPointMode),
                              onShapesPresent=(saveAs, hideAll, showAll),
                              createActions=(
                                  createRect, createPoint, createPointMode
                              ))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        addActions(self.menus.file,
                   (open, opendir, changeSavedir, openAnnotation, self.menus.recentFiles,
                    save, saveAs, close, resetAll, quit))
        addActions(self.menus.help, (help, showInfo))
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            self.displayLabelOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, saveAs, None,
            createRect, createPoint, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, save, None,
            createRect, createPointMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display slice number, for 3d image
        self.sliceNumber = QLabel('')
        self.sliceNumber.setMinimumWidth(150)
        self.statusBar().addPermanentWidget(self.sliceNumber)

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.labelCoordinates.setMinimumWidth(200)
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(silent=True)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createPointMode.setEnabled(True)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool = self.actions.beginner
        else:
            tool = self.actions.advanced
        self.tools.clear()
        addActions(self.tools, tool)
        self.menus.edit.clear()
        actions = (self.actions.createRect, self.actions.createPoint) if self.beginner()\
            else (self.actions.createRect, self.actions.createPointMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createRect.setEnabled(True)
        self.actions.createPoint.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.labelTable.setRowCount(0)      # Keep table header
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()

    def currentItem(self):
        if self.dim == TWO_D:
            items = self.labelList.selectedItems()
            if items:
                return items[0]
        else:
            items = self.labelTable.selectedItems()
            if items:
                # We only choice `label` cell for indexing
                if len(items) > 2:
                    return items[-1]
                else:   # Only `label` cell is selected
                    return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createRect(self):
        self.canvas.setEditing(False, Shape.RECTANGLE)
        self.actions.createRect.setEnabled(False)
        self.actions.createPoint.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)

    def createPoint(self):
        assert self.beginner()
        self.canvas.setEditing(False, Shape.POINT)
        self.actions.createRect.setEnabled(True)
        self.actions.createPoint.setEnabled(False)
        self.actions.createPointMode.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.createRect.setEnabled(True)
            self.actions.createPoint.setEnabled(True)

    def toggleDrawMode(self, edit=True, drawType=None):
        self.canvas.setEditing(edit, drawType)
        if edit or drawType == Shape.RECTANGLE:
            self.actions.create.setEnabled(edit)
        if edit or drawType == Shape.POINT:
            self.actions.createPointMode.setEnabled(edit)
        else:
            self.actions.createPointMode.setEnabled(not edit)

    def setCreatePointMode(self):
        assert self.advanced()
        self.toggleDrawMode(False, Shape.POINT)

    def setEdit(self):
        self.canvas.setEditing(True)
        for act in self.actions.createActions:
            act.setEnabled(True)
        self.canvas.restoreCursor()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
                if self.dim == THREE_D and shape in self.shapesToItems:
                    for item in self.shapesToOthers[shape]:
                        item.setSelected(True)
            else:
                if self.dim == TWO_D:
                    self.labelList.clearSelection()
                else:
                    self.labelTable.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape, axis=None, idx=None):
        if self.dim == TWO_D:
            shape.paintLabel = self.displayLabelOption.isChecked()
            item = HashableQListWidgetItem(shape.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setBackground(generateColorByText(shape.label))
            self.itemsToShapes[item] = shape
            self.shapesToItems[shape] = item
            self.labelList.addItem(item)
        else:
            if axis is None:
                axis = self.axis
            if idx is None:
                idx = self.idx
            shape.paintLabel = self.displayLabelOption.isChecked()
            row = self.labelTable.rowCount()
            self.labelTable.setRowCount(row + 1)
            color = generateColorByText(shape.label)
            item0 = QTableWidgetItem(str(axis))
            item0.setBackground(color)
            item0.setFlags(item0.flags() ^ Qt.ItemIsEditable)
            self.labelTable.setItem(row, 0, item0)
            item1 = QTableWidgetItem(str(idx))
            item1.setBackground(color)
            item1.setFlags(item1.flags() ^ Qt.ItemIsEditable)
            self.labelTable.setItem(row, 1, item1)

            z1 = shape.z1 if shape.z1 >= 0 else idx
            item2 = HashableQTableWidgetItem(2, str(z1))
            item2.setBackground(color)
            self.itemsToShapes[item2] = shape
            self.labelTable.setItem(row, 2, item2)
            z2 = shape.z2 if shape.z2 >= 0 else idx
            item3 = HashableQTableWidgetItem(3, str(z2))
            item3.setBackground(color)
            self.itemsToShapes[item3] = shape
            self.labelTable.setItem(row, 3, item3)
            shape.z1 = z1
            shape.z2 = z2

            item = HashableQTableWidgetItem(4, shape.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setBackground(color)
            self.itemsToShapes[item] = shape
            self.shapesToItems[shape] = item
            self.shapesToOthers[shape] = [item0, item1, item2, item3]
            self.labelTable.setItem(row, 4, item)

        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        if self.dim == TWO_D:
            self.labelList.takeItem(self.labelList.row(item))
            del self.shapesToItems[shape]
            del self.itemsToShapes[item]
        else:
            self.labelTable.removeRow(item.row())
            del self.shapesToItems[shape]
            del self.itemsToShapes[item]
            del self.shapesToOthers[shape]

    def loadLabels(self, shapes):
        shapes_3d = {}
        for case in shapes:
            if isinstance(case, (list, tuple)):
                label, points, line_color, fill_color = case
                dtype = Shape.RECTANGLE
                axis = 0
                sidx = 0
                z1 = sidx
                z2 = sidx
                fg = True
            elif isinstance(case, dict):
                dtype, label, points, line_color, fill_color = \
                    case["type"], case["label"], case["shape"], case["color1"], case["color2"]
                axis = case.get("axis", 0)
                sidx = case.get("slice", 0)
                fg = case.get("fg", True)
                z1 = case.get("z1", sidx)
                z2 = case.get("z2", sidx)
            else:
                raise RuntimeError("Wrong shape:", case)
            
            if dtype == Shape.RECTANGLE:
                shape = Shape(label=label)
                shape.z1 = z1
                shape.z2 = z2
            elif dtype == Shape.POINT:
                shape = Point(label=label, foreground=fg)
            else:
                raise TypeError("Not recognized shape type:", dtype)
            
            for x, y in points:
                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.close()

            key = DSKey.ds2key(axis, sidx)
            if key not in shapes_3d:
                shapes_3d[key] = []
            shapes_3d[key].append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)

            self.addLabel(shape, axis, sidx)

        self.canvas.loadShapes(shapes_3d)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s, key=None):
            items = self.shapesToOthers[s]
            formatted = dict(label=s.label,
                             dtype=s.type_,
                             line_color=s.line_color.getRgb(),
                             fill_color=s.fill_color.getRgb(),
                             points=[(p.x(), p.y()) for p in s.points],
                             z1=int(items[2].text()),
                             z2=int(items[3].text()),
                             )
            if key:
                formatted["axis"] = key[0]
                formatted["slice"] = key[1]
                if s.type_ == Shape.POINT:
                    formatted['fg'] = s.fg
            return formatted

        if self.dim == TWO_D:
            shapes = [format_shape(shape) for shape in self.canvas.shapes_3d[DSKey.ds2key(0, 0)]]
        else:
            shapes = [format_shape(shape, DSKey.key2ds(key))
                      for key, shapes in self.canvas.shapes_3d.items() for shape in shapes]
        # Can add differrent annotation formats here
        try:
            if annotationFilePath[-5:].lower() != ".cxml":
                annotationFilePath += COMM_EXT
            self.labelFile.saveCommonFormat(annotationFilePath, shapes, self.filePath)
            print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        shape = self.canvas.copySelectedShape()
        if shape:
            self.addLabel(shape)
            # fix copy and delete
            self.shapeSelectionChanged(True)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if not isinstance(item, (HashableQTableWidgetItem, HashableQListWidgetItem)):
            return
        if item and self.canvas.editing():
            self._noSelectionSlot = True    # Avoid selection loop between shape <-> item
            shape = self.itemsToShapes[item]
            if self.dim != TWO_D:
                slice_index = int(self.shapesToOthers[shape][1].text())
                self.idx = slice_index
                self.canvas.setPtr(self.axis, self.idx)
                self.canvas.deSelectShape()
                self.updateCanvasImage()
            self.canvas.selectShape(shape)

    def labelItemChanged(self, item):
        if not isinstance(item, (HashableQTableWidgetItem, HashableQListWidgetItem)):
            return
        shape = self.itemsToShapes[item]

        if item.col_ == 4 and item.text() != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        elif item.col_ == 2:
            shape.z1 = int(item.text())
        elif item.col_ == 3:
            shape.z2 = int(item.text())
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelShowStateChanged(self):
        state = self.labelShowCheckBox.checkState()
        for i in range(self.labelTable.rowCount()):
            self.labelTable.item(i, 4).setCheckState(state)

    # Callback functions:
    def newRect(self):
        text = "rectangle"
        generate_color = generateColorByText(text)
        shape = self.canvas.setLastLabel(text, generate_color, generate_color)
        self.addLabel(shape)
        self.canvas.setEditing(True)
        for act in self.actions.createActions:
            act.setEnabled(True)
        self.setDirty()

    def newPoint(self):
        self.segAlgButtonRun.setEnabled(True)
        if self.segAutoCheckBox.isChecked():
            self.segRun()
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                for act in self.actions.createActions:
                    act.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation, mode):
        """
        mode 0: only scroll inside image
             1: 
                TWO_D: the same as 0
                THREE_D: slice scroll
        """
        if mode == 0 or self.dim == TWO_D:
            units = - delta / (8 * 15)
            bar = self.scrollBars[orientation]
            bar.setValue(bar.value() + bar.singleStep() * units)
        else:   # THREE_D
            self.idx = self.i3d.check(self.idx, delta // 120)
            self.canvas.setPtr(self.axis, self.idx)
            self.canvas.deSelectShape()
            self.updateCanvasImage()

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(int(value))

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + int(increment))

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(int(new_h_bar_value))
        v_bar.setValue(int(new_v_bar_value))

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)

        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()
                
        if unicodeFilePath and os.path.exists(unicodeFilePath):
            # Load image:
            # read data first and store for saving into label file.
            if not self.is3dimage(unicodeFilePath):
                self.imageData = read(unicodeFilePath, None)
                self.dim = TWO_D
            else:
                self.i3d = read3d(unicodeFilePath)
                self.i3d.setIntensityClip(low=self.int_low.value(), high=self.int_high.value())
                self.dim = THREE_D
                self.ref = self.i3d.label
            self.labelFile = None
            self.canvas.verified = False

            if self.dim == TWO_D:
                self.sliceNumber.setText("")
                image = QImage.fromData(self.imageData)
                self.labelList.setVisible(True)
                self.labelTable.setVisible(False)
                self.settingDock.setVisible(False)
            else:
                self.idx = 0
                self.sliceNumber.setText("Slice: {:3d} / Total: {:3d}".format(self.idx, self.i3d.shape[self.axis]))
                image = self.i3d.at(self.idx, self.axis,
                                    self.segShowCheckBox.isChecked())
                self.labelList.setVisible(False)
                self.labelTable.setVisible(True)
                self.settingDock.setVisible(True)
                self.segAlgButtonRun.setEnabled(True)
                self.segAlgButtonSave.setEnabled(True)
                self.total_time = 0
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            if self.defaultSaveDir is not None:
                basename = os.path.basename(os.path.splitext(self.filePath)[0])
                if "." in basename:
                    basename = basename.split(".")[0]
                cxmlPath = os.path.join(self.defaultSaveDir, basename + COMM_EXT)
                
                """Annotation file priority: Common """
                if os.path.isfile(cxmlPath):
                    print("Load annotation: ", cxmlPath)
                    self.loadCommonXMLByFilename(cxmlPath)
                else:
                    self.canvas.loadShapes({})
            else:
                cxmlPath = os.path.splitext(filePath)[0] + COMM_EXT
                if os.path.isfile(cxmlPath):
                    print("Load annotation: ", cxmlPath)
                    self.loadCommonXMLByFilename(cxmlPath)
                else:
                    self.canvas.loadShapes({})

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.dim == TWO_D and self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count()-1))
                self.labelList.item(self.labelList.count()-1).setSelected(True)
            if self.dim == THREE_D and self.labelTable.rowCount():
                self.labelTable.setCurrentItem(self.labelTable.item(self.labelTable.rowCount() - 1, 4))
                self.labelTable.item(self.labelTable.rowCount() - 1, 4).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''
        if self.defaultSegDir and os.path.exists(self.defaultSegDir):
            settings[SETTING_SEG_DIR] = ustr(self.defaultSegDir)
        else:
            settings[SETTING_SEG_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()

        settings[SETTING_AUTO_SEG] = self.segAutoCheckBox.isChecked()
        settings[SETTING_ALPHA] = self.segResSlider.value()
        settings[SETTING_INT_MIN] = self.int_low.value()
        settings[SETTING_INT_MAX] = self.int_high.value()
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        # Priorities: 3d > 2d
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        extensions_3d = self.get3dformats()
        images = []
        images_3d = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
                elif file.lower().endswith(tuple(extensions_3d)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images_3d.append(path)
        natural_sort(images, key=lambda x: x.lower())
        natural_sort(images_3d, key=lambda x: x.lower())
        return images + images_3d

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(
            self, '%s - Save annotations to the directory' % __appname__, path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotationDialog(self, _value=False):
        # if self.filePath is None:
        #     self.statusBar().showMessage('Please select image first')
        #     self.statusBar().show()
        #     return
        #
        # path = os.path.dirname(ustr(self.filePath))\
        #     if self.filePath else '.'
        # if self.usingPascalVocFormat:
        #     filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
        #     filename = ustr(QFileDialog.getOpenFileName(self,'%s - Choose a xml file' % __appname__, path, filters))
        #     if filename:
        #         if isinstance(filename, (tuple, list)):
        #             filename = filename[0]
        #     self.loadPascalXMLByFilename(filename)
        pass

    def openDirDialog(self, _value=False, silent=False):
        if not self.mayContinue():
            return

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent != True :
            targetDirPath = ustr(QFileDialog.getExistingDirectory(
                self, '%s - Open Directory' % __appname__, defaultOpenDirPath,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)

        self.importDirImages(targetDirPath)

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters_2d = "Image files (%s)" % ' '.join(formats)
        filters_3d = "Medical image (%s)" % ' '.join(['*%s' % x for x in self.get3dformats()])
        filters = ";;".join([filters_3d, filters_2d])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename[0]:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        """ save labels """
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(os.path.splitext(imgFileName)[0])[0]
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(os.path.splitext(imgFileName)[0])[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog(removeExt=False))

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        annoPath = self.saveFileDialog()
        if annoPath:
            self.defaultSaveDir = os.path.dirname(annoPath)
            self._saveFile(annoPath)

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'CXML (*%s)' % LabelFile.suffix
        openDialogPath = self.defaultSaveDir
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = self.filePath.split(".")[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0]  # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadCommonXMLByFilename(self, cxmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(cxmlPath) is False:
            return

        tCommonParserReader = CommonReader(cxmlPath)
        shapes = tCommonParserReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tCommonParserReader.verified

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())

    def get3dformats(self):
        return ['.nii', '.nii.gz']

    def is3dimage(self, filePath):
        all3dformats = self.get3dformats()
        if filePath.lower().endswith(tuple(all3dformats)):
            return True
        return False

    def getPixel(self, x, y):
        if 0 <= x < self.image.width() and 0 <= y < self.image.height():
            if self.dim == TWO_D:
                c = self.image.pixelColor(x, y)
                return c.red(), c.green(), c.blue()
            else:
                return [self.i3d.pixel(self.idx, int(y), int(x), self.axis)]
        else:
            return 0, 0, 0

    def updateCanvasImage(self, image=None):
        if isinstance(image, np.ndarray):
            self.image = image
        else:
            self.image = self.i3d.at(self.idx, self.axis, self.segShowCheckBox.isChecked(),
                                     self.gtShowCheckBox.isChecked())
        self.sliceNumber.setText("Slice: {:3d} / Total: {:3d}".format(self.idx, self.i3d.shape[self.axis]))
        self.canvas.loadPixmap(QPixmap.fromImage(self.image))

    def update3dImageLow(self, value):
        if self.i3d and isinstance(value, int):
            self.i3d.setIntensityClip(low=value)
            self.updateCanvasImage()

    def update3dImageHigh(self, value):
        if self.i3d and isinstance(value, int):
            self.i3d.setIntensityClip(high=value)
            self.updateCanvasImage()

    def segRun(self):
        segAlg = self.segAlgComboBox.currentText()
        if self.i3d is not None:
            start = time.time()
            name = self.segName[self.segAlg.index(segAlg)]
            centers = {"fg": [], "bg": []}
            stddevs = {"fg": [], "bg": []}
            bbox = [None] * 6
            for key, shapes in self.canvas.shapes_3d.items():
                for shape in shapes:
                    if not self.canvas.isVisible(shape):
                        continue
                    if shape.type_ == Shape.RECTANGLE:
                        x1, y1, w, h = shape.rect()
                        bbox[0] = int(shape.z1)
                        bbox[1] = int(y1)
                        bbox[2] = int(x1)
                        bbox[3] = int(shape.z2) + 1
                        bbox[4] = int(y1 + h)
                        bbox[5] = int(x1 + w)
                    if shape.type_ == Shape.POINT:
                        if shape.fg:
                            k = "fg"
                            stddevs[k].append([2., self.stddev.value(), self.stddev.value()])
                        else:
                            k = "bg"
                            stddevs[k].append([1., self.stddev.value(), self.stddev.value()])
                        centers[k].append([int(DSKey.key2ds(key)[1]), int(shape.points[0].y()), int(shape.points[0].x())])
            if segAlg == "Test":
                success = self.i3d.seg_test(bbox, centers, stddevs)
            elif segAlg == "DIN":
                success = self.i3d.seg_din(bbox, centers, stddevs, name, guide_type="exp")
            elif segAlg == "EDT":
                success = self.i3d.seg_din(bbox, centers, stddevs, name, guide_type="euc")
            elif segAlg == "GDT":
                success = self.i3d.seg_din(bbox, centers, stddevs, name, guide_type="geo")
            elif segAlg == "Random Walk 3D":
                success = self.i3d.seg_RW(bbox, centers)
            elif segAlg == "Graph Cut 3D":
                success = self.i3d.seg_GraphCut(bbox, centers)
            else:
                success = 1
            diff = time.time() - start
            if success == 0:
                self.updateCanvasImage()
                self.segAlgButtonUndo.setEnabled(True)
                self.segAlgButtonClear.setEnabled(True)
                self.total_time += diff
            else:
                QMessageBox.critical(self, "Error", "Please select correct segmentation method.", QMessageBox.Yes)

    def segClear(self):
        if self.i3d is not None and self.idx in self.i3d.segCache:
            self.i3d.segCache.pop(self.idx)
            self.updateCanvasImage()
        elif self.i3d is not None and isinstance(self.i3d.segCache, np.ndarray):
            self.i3d.backup = self.i3d.segCache
            self.i3d.segCache = {}
            self.segAlgButtonClear.setEnabled(False)
            self.updateCanvasImage()

    def segUndo(self):
        if self.i3d is not None:
            res = self.i3d.undo()
            if res:
                self.segAlgButtonUndo.setEnabled(False)
                self.updateCanvasImage()

    def segSave(self):
        if self.i3d is not None and len(self.i3d.segCache) > 0:
            if isinstance(self.i3d.segCache, np.ndarray) or self.idx in self.i3d.segCache:
                # Count interaction numbers
                # count = 0
                # for i in range(self.labelTable.rowCount()):
                #     idx = int(self.labelTable.item(i, 1).text())
                #     if idx == self.idx:
                #         count += 1
                saveFile = self.saveSegDialog("", removeExt=False)
                if saveFile:
                    self.defaultSegDir = os.path.dirname(saveFile)
                    if isinstance(self.i3d.segCache, dict):
                        seg = np.zeros(self.i3d.shape, dtype=np.uint8)
                        for i in self.i3d.segCache:
                            seg[i] = self.i3d.segCache[i]
                    else:
                        seg = self.i3d.segCache
                    write_nii(seg, self.i3d.meta.meta, saveFile)

    def saveSegDialog(self, suffix, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'Segmentation (*.nii.gz)'
        openDialogPath = self.currentPath() if self.defaultSegDir is None else self.defaultSegDir
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix("nii.gz")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.basename(self.filePath).split(".")[0] + suffix
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0]  # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def segResModeChanged(self):
        if self.i3d is None:
            return
        if self.segResMode.text() == "Mask":
            self.segResMode.setText("Contour")
            self.i3d.contour = True
            self.updateCanvasImage()
        else:
            self.segResMode.setText("Mask")
            self.i3d.contour = False
            self.updateCanvasImage()

    def segResColorPicker(self):
        color = QColorDialog.getColor()
        if sum(color.getRgb()[:3]) == 0:
            return
        if self.i3d is not None:
            self.segResColor.setStyleSheet("background-color: %s; color: white" % color.name())
            self.i3d.color = color.getRgb()[:3]
            self.updateCanvasImage()

    def segResColorPickerGT(self):
        color = QColorDialog.getColor()
        if sum(color.getRgb()[:3]) == 0:
            return
        if self.i3d is not None:
            self.segGTColor.setStyleSheet("background-color: %s; color: white" % color.name())
            self.i3d.colorGT = color.getRgb()[:3]
            self.updateCanvasImage()

    def segResSliderChanged(self):
        if self.i3d is not None:
            value = self.segResSlider.value() / 100
            self.i3d.alpha = value
            self.updateCanvasImage()

    def computeDiceClicked(self):
        if len(self.i3d.segCache) == 0:
            return
        if self.ref is None:
            ref_file = os.path.join(os.path.dirname(self.filePath),
                                    os.path.basename(self.filePath).replace("img", "mask")
                                    .replace("volume", "segmentation"))
            header, self.ref = read_nii(ref_file, out_dtype=np.uint8)

        # get bbox
        bbox = (slice(None), slice(None), slice(None))
        find = False
        for _, shapes in self.canvas.shapes_3d.items():
            for shape in shapes:
                if shape.type_ == Shape.RECTANGLE and self.canvas.isVisible(shape):
                    x1, y1, w, h = shape.rect()
                    bbox = (slice(int(shape.z1), int(shape.z2 + 1)),
                            slice(int(y1), int(y1 + h)),
                            slice(int(x1), int(x1 + w)))
                    find = True
                    break
            if find:
                break
        dice, vd, rvd = computeMetrics(self.ref, self.i3d.segCache, bbox, self.i3d.unit)
        QMessageBox.information(self, "Info", f"{os.path.basename(self.filePath)}\n\nDice: {dice}\nVD: {vd:.0f}\nRVD: {rvd * 100:.2f}",
                                QMessageBox.Yes)

    # def computeDiceStateChanged(self):
    #     if self.segComputeDiceCheckBox.isChecked():
    #         self.canvas.overrideCursor(Qt.PointingHandCursor)
    #     else:
    #         self.canvas.overrideCursor(Qt.ArrowCursor)


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile saveDir
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(
                         os.path.dirname(sys.argv[0]),
                         'data', 'medical.txt'),
                     argv[3] if len(argv) >= 4 else None)
    win.show()
    return app, win


def main():
    """construct main app and run it"""
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
