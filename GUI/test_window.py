from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout
import pyqtgraph as pg
from Algorithm.PreProcess import norm
import numpy as np
from Algorithm.NeuralNetwrok import pred


class Load_Model_Worker(QObject):
    finished = pyqtSignal()
    resultReady = pyqtSignal(object)

    def __init__(self, model):
        super().__init__()
        self.model = model

    @pyqtSlot(np.ndarray)
    def load_model(self, img):
        prediction = pred(patch=img, model=self.model)
        self.resultReady.emit(prediction)
        self.finished.emit()


class TestWindow(QMainWindow):
    def __init__(self, x, y, image, t, model):
        super().__init__()

        self.image = norm(image.copy(), costume_min=0, costume_max=t)

        y = np.clip(y, 150, image.shape[0]-151)
        x = np.clip(x, 150, image.shape[1]-151)

        self.image = self.image[y-150:y+150, x-150:x+150]
        self.model = model

        self.initUI()

    def initUI(self):
        # Main widget and layout
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)

        # Images layout
        self.imagesLayout = QHBoxLayout()
        self.layout.addLayout(self.imagesLayout)

        # Plot widgets
        self.plotWidget1 = pg.PlotWidget()
        self.plotWidget2 = pg.PlotWidget()

        # Image items
        self.imageItem1 = pg.ImageItem(self.image, levels=(0, 255))
        self.imageItem2 = pg.ImageItem(self.image, levels=(0, 255))  # Flip the image horizontally

        self.imageItem2.setRotation(-90)
        self.imageItem1.setRotation(-90)

        # Adding Image items to Plot widgets
        self.plotWidget1.addItem(self.imageItem1)
        self.plotWidget2.addItem(self.imageItem2)

        # Setting up shared axes
        self.plotWidget2.setXLink(self.plotWidget1)
        self.plotWidget2.setYLink(self.plotWidget1)

        # Adding Plot widgets to the images layout
        self.imagesLayout.addWidget(self.plotWidget1)
        self.imagesLayout.addWidget(self.plotWidget2)

        self.pred()

    def pred(self):
        self.thread = QThread()
        self.worker = Load_Model_Worker(self.model)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(lambda: self.worker.load_model(self.image))
        self.worker.resultReady.connect(self.handle_result)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @pyqtSlot(object)
    def handle_result(self, prediction):
        self.imageItem2.setImage(prediction)
        self.imageItem2.setLevels([0, 1])
