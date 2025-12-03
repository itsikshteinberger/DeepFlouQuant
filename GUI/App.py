import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from PyQt5.QtCore import Qt, QSize, QObject, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QMovie, QImage, QPixmap, QWheelEvent, QPainter, QMouseEvent
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget, \
    QSizePolicy, QSlider, QPushButton, QGraphicsPixmapItem, QGraphicsView, QGraphicsScene, QRadioButton
from test_window import TestWindow
from Algorithm import PreProcess, PostProcess, NeuralNetwrok
import visualization_tools as vis
from pyqtgraph import PlotWidget, ImageItem, ScatterPlotItem, LegendItem
import pandas as pd
import numpy as np
import cv2


MODEL = None

colors = ['cyan', 'pink', 'yellow', 'gold', 'lightcoral', 'lightgreen', 'lightskyblue', 'mediumorchid', 'lightsalmon']


class Load_Model_Worker(QObject):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    @pyqtSlot()
    def load_model(self):
        global MODEL
        MODEL = NeuralNetwrok.load_model()
        self.finished.emit()


class Process_Worker(QObject):
    finished = pyqtSignal()
    status_string = pyqtSignal(str)
    all_data_ready = pyqtSignal(dict)

    def __init__(self, data, thresholds):
        super().__init__()
        self.images = [img.copy() for img in data['images']]
        self.names = data['names']
        self.thresholds = thresholds
        self.dapi_img = data['dapi']

    @pyqtSlot()
    def process_images(self):
        global MODEL
        pipeline_images = {}

        self.status_string.emit('Initiating image processing')
        images = PreProcess.multi_norm(images=self.images, thresholds=self.thresholds)
        # images = [PreProcess.norm(img, costume_min=0, costume_max=t) for img, t in zip(self.images, self.thresholds)]

        pipeline_images['preprocess'] = [img.copy() for img in images]

        self.status_string.emit('Loading model for analysis')
        model = NeuralNetwrok.load_model() if not MODEL else MODEL
        MODEL = model

        for i, img in enumerate(images):
            self.status_string.emit(f'Analyzing image {i + 1} of {len(images)}')
            images[i] = NeuralNetwrok.modelEval(img, model=model)
        pipeline_images['ann'] = [img.copy() for img in images]

        dataframes = []
        for i, img in enumerate(images):
            self.status_string.emit(f'Extracting blobs from image {i + 1} of {len(images)}')
            dataframes.append(PostProcess.get_df(image=img))

        self.status_string.emit('Removing noise from the data')
        dataframes = PostProcess.clean_noise(dataframes=dataframes, img=images[0], s=300, min_t=1)
        pipeline_images['postprocess'] = dataframes

        self.status_string.emit('Creating reference image')
        nis_img = vis.drawNis(self.images)

        self.status_string.emit('Segmenting the Dapi channel')
        dapi = PreProcess.process_dapi(self.dapi_img)
        dapi['img'] = vis.draw_border(dapi)

        self.status_string.emit('Combining the results')
        result_df = PostProcess.count(dapi=dapi, names=self.names, dataframes=dataframes)

        self.status_string.emit('Image processing completed')

        all_data = {
            'dataframes': dataframes,
            'result_df': result_df,
            'dapi_img': dapi['img'],
            'nis_img': nis_img,
            'pipeline': pipeline_images
        }

        self.all_data_ready.emit(all_data)

        self.finished.emit()


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.zoom_factor = 1.25  # Adjust this value to change zoom sensitivity
        self.current_zoom = 0

        self.open_windows = []

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            zoom_in_factor = self.zoom_factor
            self.current_zoom += 1
        else:
            zoom_in_factor = 1 / self.zoom_factor
            self.current_zoom -= 1

        self.scale(zoom_in_factor, zoom_in_factor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        scene_pos = event.pos()
        image_item = self.main_window.pixmap_item
        image_rect = image_item.boundingRect()

        if image_rect.contains(scene_pos):
            mapped_pos = self.mapToScene(scene_pos)
            image_pos = image_item.mapFromScene(mapped_pos)

            x = int(image_pos.x())
            y = int(image_pos.y())

            new_window = TestWindow(
                x=x,
                y=y,
                image=self.main_window.data['images'][self.main_window.current_index],
                t=self.main_window.thresholds[self.main_window.current_index],
                model=MODEL
            )
            new_window.show()
            self.open_windows.append(new_window)

        super().mouseDoubleClickEvent(event)


class Application(QMainWindow):
    def __init__(self):
        super().__init__()

        self.createMenuBar()
        self.loading_screen(title='Loading model')
        self.load_model()
        self.showMaximized()

        self.data = {}
        self.current_index = 0

    def fileSelector(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select TIFF File", "", "TIFF Files (*.tif *.tiff)", options=options
        )

        if file_name:
            names, images = PreProcess.readImg(path=file_name)
            #images = [img[500: 1200, 500: 1200] for img in images]
            dapi, images, names = PreProcess.get_dapi(names=names, images=images)

            self.data = {
                'dapi': dapi,
                'names': names,
                'images': images
            }

            self.thresholds = [np.max(img) for img in images]

            self.images_screen()

    def createMenuBar(self):
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu("&File")
        newAction = fileMenu.addAction("&new")
        newAction.triggered.connect(self.fileSelector)
        newAction.setEnabled(False)

        self.menu_bar_actions = {'new': newAction}

        saveMenu = menuBar.addMenu("&Save")

        self.csvAction = saveMenu.addAction("&Result CSV")
        self.csvAction.triggered.connect(lambda: self.save(file=self.result['result_df']))
        self.csvAction.setEnabled(False)

        self.orgAction = saveMenu.addAction("&Original images")
        self.orgAction.triggered.connect(lambda: self.save(file=self.result['pipeline']['preprocess'], key='org_'))
        self.orgAction.setEnabled(False)

        self.annAction = saveMenu.addAction("&Processed images")
        self.annAction.triggered.connect(lambda: self.save(file=self.result['pipeline']['ann'], key='ann_'))
        self.annAction.setEnabled(False)

        self.postAction = saveMenu.addAction("&Processed CSV files")
        self.postAction.triggered.connect(lambda: self.save(file=self.result['pipeline']['postprocess'], key='post_'))
        self.postAction.setEnabled(False)

        menuBar.setStyleSheet("""
                        QMenuBar {
                            font-size: 18px; 
                            padding-right: 40px; 
                        }
                        QMenu {
                            font-size: 18px; 
                    """)

    def load_model(self):
        self.obj = Load_Model_Worker()
        self.thread = QThread()

        self.obj.moveToThread(self.thread)
        self.obj.finished.connect(self.thread.quit)
        self.thread.started.connect(self.obj.load_model)
        self.thread.finished.connect(self.images_screen)

        self.thread.start()

    def images_screen(self):
        self.clear_screen()
        self.menu_bar_actions['new'].setEnabled(True)  # Uncomment if you have menu bar actions

        if 'images' not in self.data.keys():
            return

        central_widget = QWidget()
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # Create Graphics View and Scene
        self.graphics_view = ZoomableGraphicsView(self, self)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.graphics_scene)

        # Create bottom layout
        bottom_layout = QHBoxLayout()

        # Create slider status by slider
        self.status_label = QLabel()

        # Create slider
        self.t_slider = QSlider()
        self.t_slider.setOrientation(Qt.Horizontal)
        self.t_slider.valueChanged.connect(self.t_change)
        self.set_slider()

        # Create buttons
        self.prev_button = QPushButton('<')
        self.next_button = QPushButton('>')

        self.prev_button.setDisabled(False)
        self.current_index = 0

        self.prev_button.setDisabled(False)

        self.prev_button.clicked.connect(lambda: self.change_index(-1))
        self.next_button.clicked.connect(lambda: self.change_index(1))

        bottom_layout.addWidget(self.prev_button)
        bottom_layout.addWidget(self.t_slider)
        bottom_layout.addWidget(self.status_label)
        bottom_layout.addWidget(self.next_button)

        self.set_image()
        self.set_slider()

        # Add graphics view and bottom layout to main layout
        main_layout.addWidget(self.graphics_view, stretch=1)
        main_layout.addLayout(bottom_layout)

        # Set main layout for central widget
        central_widget.setLayout(main_layout)

    def t_change(self, value):
        self.thresholds[self.current_index] = value
        self.status_label.setText(f'{self.data["names"][self.current_index]} | {self.thresholds[self.current_index]}')
        self.set_image()

    def set_image(self):
        image_array = self.data['images'][self.current_index].copy()
        image_array = PreProcess.norm(image_array, costume_min=0, costume_max=self.thresholds[self.current_index])
        height, width = image_array.shape
        qimage = QImage(image_array.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.graphics_scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.graphics_scene.addItem(self.pixmap_item)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_slider(self):
        image_array = self.data['images'][self.current_index].copy()
        self.t_slider.setMinimum(np.min(image_array) + 1)
        self.t_slider.setMaximum(np.max(image_array) * 5)
        self.t_slider.setValue(self.thresholds[self.current_index])
        self.status_label.setText(f'{self.data["names"][self.current_index]} | {self.thresholds[self.current_index]}')

    def change_index(self, change):
        self.current_index += change

        if self.current_index >= (len(self.data['images']) - 1):
            self.current_index = len(self.data['images']) - 1
            if self.next_button.text() == 'Done':
                self.preProcess()
            else:
                self.next_button.setText('Done')
        else:
            self.next_button.setText('>')

        if self.current_index <= 0:
            self.current_index = 0
            self.prev_button.setDisabled(True)
        else:
            self.prev_button.setDisabled(False)

        self.set_slider()
        self.set_image()

    def preProcess(self):
        self.clear_screen()

    def clear_screen(self):
        self.setCentralWidget(QWidget())

    def loading_screen(self, title=''):
        # Create the main vertical layout
        loading_layout = QVBoxLayout()
        loading_layout.setAlignment(Qt.AlignCenter)
        loading_layout.setContentsMargins(50, 50, 50, 50)  # Add some padding
        loading_layout.setSpacing(60)

        # Create a horizontal layout for the GIF
        movie_layout = QHBoxLayout()
        movie_layout.setAlignment(Qt.AlignCenter)

        movie = QMovie("loading.gif")
        movie.setScaledSize(QSize(150, 150))
        loading_movie_label = QLabel()
        loading_movie_label.setMovie(movie)
        movie.start()

        movie_layout.addWidget(loading_movie_label)

        # Create a horizontal layout for the status label
        label_layout = QHBoxLayout()
        label_layout.setAlignment(Qt.AlignCenter)

        self.loading_label = QLabel(title)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 10pt; color: #333;")

        label_layout.addWidget(self.loading_label)

        # Add the movie and label layouts to the main layout
        loading_layout.addLayout(movie_layout)
        loading_layout.addLayout(label_layout)

        central_widget = QWidget()
        central_widget.setLayout(loading_layout)
        self.setCentralWidget(central_widget)

    def preProcess(self):
        self.clear_screen()
        self.loading_screen()
        self.freeze_menuBar()

        self.obj = Process_Worker(self.data, self.thresholds)
        self.thread = QThread()

        self.obj.status_string.connect(lambda i: self.loading_label.setText(i))
        self.obj.moveToThread(self.thread)
        self.obj.finished.connect(self.thread.quit)
        self.thread.started.connect(self.obj.process_images)
        self.obj.all_data_ready.connect(self.receive_all_data)
        self.thread.finished.connect(self.result_screen)

        self.thread.start()

    def freeze_menuBar(self):
        for action in self.menuBar().actions():
            action.setEnabled(False)

    def unfreeze_menuBar(self):
        for action in self.menuBar().actions():
            action.setEnabled(True)

    def receive_all_data(self, all_data):
        self.result = all_data

    def result_screen(self):
        self.clear_screen()
        self.unfreeze_menuBar()
        self.csvAction.setEnabled(True)
        self.orgAction.setEnabled(True)
        self.annAction.setEnabled(True)
        self.postAction.setEnabled(True)

        background_img = self.result['dapi_img']
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Create first row layout (plots)
        self.plots_layout = QHBoxLayout()
        self.layout.addLayout(self.plots_layout)

        # Create PlotWidget for scatter plot
        self.scatter_plot = PlotWidget()
        self.plots_layout.addWidget(self.scatter_plot)

        # Set fixed aspect ratio policy
        self.scatter_plot.setAspectLocked(True)

        # Plot scatter data
        x, y = self.result['result_df']['x_center'], self.result['result_df']['y_center']
        self.scatter_plot.clear()
        img_item = ImageItem(background_img)
        self.scatter_plot.addItem(img_item)
        img_item.setRotation(-90)
        self.legend = LegendItem()
        self.scatter_plot.addItem(self.legend, (1, 1))

        data = self.result['result_df'].apply(lambda row: '\n' + '\n'.join(
            [f"{col_name}: {int(col_value)}" for col_name, col_value in row.iloc[2:].items()]), axis=1)
        scatter_plot_data = ScatterPlotItem(x=x, y=y * (-1), data=data, brush='b', hoverable=True, name='center')

        for i, df in enumerate(self.result['dataframes']):
            sp = ScatterPlotItem(x=df['x'], y=(-1) * df['y'], brush=colors[i], name=self.data['names'][i])
            self.scatter_plot.addItem(sp)
            self.legend.addItem(sp, self.data['names'][i])

        self.scatter_plot.sigXRangeChanged.connect(self.update_scatter_size)
        self.scatter_plot.sigYRangeChanged.connect(self.update_scatter_size)
        self.scatter_plot.addItem(scatter_plot_data)

        # Create second row layout (button)
        button_layout = QHBoxLayout()
        self.layout.addLayout(button_layout)

        # Create radio buttons for each dataframe
        self.radio_buttons = []
        radio_button = QRadioButton('all')
        radio_button.setChecked(True)
        button_layout.addWidget(radio_button)
        self.radio_buttons.append(radio_button)

        for i, df in enumerate(self.result['dataframes']):
            radio_button = QRadioButton(self.data['names'][i])
            button_layout.addWidget(radio_button)
            self.radio_buttons.append(radio_button)

        for rb in self.radio_buttons:
            rb.toggled.connect(lambda checked: self.toggle_dataframe(checked))

        self.show_nis_img = False
        self.toggle_plot_button = QPushButton('Show reference image')
        self.toggle_plot_button.clicked.connect(self.toggle_nis_img)
        button_layout.addWidget(self.toggle_plot_button)

    def save(self, file, key=''):
        if isinstance(file, pd.DataFrame):
            file_name, _ = QFileDialog.getSaveFileName(self, "Save DataFrame to CSV", "", "CSV Files (*.csv)")
            if file_name:
                file.to_csv(file_name, index=False)

        elif isinstance(file, list):
            if isinstance(file[0], pd.DataFrame):
                save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save DataFrames")
                for i, df in enumerate(file):
                    file_path = save_dir + fr'/{key + self.data["names"][i]}.csv'
                    df.to_csv(file_path, index=False)
            if isinstance(file[0], np.ndarray):
                save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save DataFrames")
                for i, img in enumerate(file):
                    img_path = save_dir + fr'/{key + self.data["names"][i]}.png'
                    cv2.imwrite(img_path, PreProcess.norm(img))

    def toggle_dataframe(self, checked):
        rb_text = ''
        for rb in self.radio_buttons:
            if rb.isChecked():
                rb_text = rb.text()
                break

        for scatter_plot in self.scatter_plot.items():
            if isinstance(scatter_plot, ScatterPlotItem):
                if rb_text == scatter_plot.name() or rb_text == 'all' or scatter_plot.name() == 'center':
                    scatter_plot.setPointsVisible(True)
                else:
                    scatter_plot.setPointsVisible(False)

    def toggle_nis_img(self):
        if self.show_nis_img:
            self.plots_layout.removeWidget(self.nis_plot)
            self.nis_plot.close()
            self.show_nis_img = False
            self.toggle_plot_button.setText('Show reference image')
        else:
            self.nis_plot = PlotWidget()

            img_item = ImageItem(self.data['dapi'])
            self.nis_plot.addItem(img_item)
            img_item.setRotation(-90)
            for i, img in enumerate(self.result['nis_img']):
                img_item = ImageItem(img)
                self.nis_plot.addItem(img_item)
                if i > 0:
                    img_item.setOpacity(0.7)
                else:
                    img_item.setOpacity(1)
                img_item.setRotation(-90)

            self.nis_plot.setXLink(self.scatter_plot)
            self.nis_plot.setYLink(self.scatter_plot)

            self.plots_layout.addWidget(self.nis_plot)
            self.toggle_plot_button.setText('Hide reference image')
            self.show_nis_img = True

    def update_scatter_size(self):
        x_range = self.scatter_plot.viewRange()[0]
        y_range = self.scatter_plot.viewRange()[1]

        zoom_level = abs(x_range[1] - x_range[0]) + abs(y_range[1] - y_range[0])

        point_size = 10000 / zoom_level
        if point_size > 19:
            point_size = 19
        if point_size < 7:
            point_size = 7

        for item in self.scatter_plot.items():
            if isinstance(item, ScatterPlotItem):
                item.setSize(point_size)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Application()
    window.show()
    sys.exit(app.exec_())
