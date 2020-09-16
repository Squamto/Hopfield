import numpy
import sys
from datetime import datetime

import PyQt5

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtOpenGL

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *

from hopfield_util import HopfieldNetwork, MNISTHandler, load_example

'''
useOpenGl = True
try:
    from OpenGL import GL
except ImportError:
    app = QApplication(sys.argv)
    messageBox = QMessageBox(QMessageBox.Warning, "Hopfield", "PyOpenGL is not installed. Running without Hardware acceleration.", QMessageBox.Close)
    messageBox.setDetailedText("Run:\npip install PyOpenGL PyOpenGL_accelerate")
    messageBox.exec_()
    useOpenGL = False
'''

class GridHelper:
    def __init__(self, width : int, height : int, background_color : QColor = Qt.black, show_marker = False):
        self.width = width
        self.height= height
        self.show_marker = show_marker
        self.background = QBrush(background_color)
        self.marker_size = 0.3
        
        self.w = -1
        self.h = -1
        self.x0 = -1
        self.y0 = -1
        self.xl = 0
        self.yl = 0
        self.rect_size = -1

    def calculate_values(self):
        self.rect_size = int(min(self.width // self.w, self.height // self.h))
        self.y0 = int((self.height - self.rect_size * self.h) / 2)
        self.x0 = int((self.width - self.rect_size * self.w) / 2)
        self.xl = self.x0 + self.rect_size*self.w
        self.yl = self.y0 + self.rect_size*self.h

    def resize(self, newSize):
        self.width = newSize.width()
        self.height = newSize.height()
        self.calculate_values()

    def draw(self, painter : QPainter, rect, array : numpy.ndarray, current_pos = (-1, -1)):
        # painter.fillRect(rect, self.background)
        self.w, self.h = array.shape
        self.calculate_values()
        if self.w <= self.width and self.h <= self.height:
            for y in range(self.h):
                for x in range(self.w):
                    painter.fillRect(self.x0 + x*self.rect_size, self.y0 + y*self.rect_size, self.rect_size, self.rect_size, Qt.black if array[y, x] == -1 else Qt.white)
            if self.show_marker:
                x, y = current_pos
                painter.fillRect(self.x0 + (x + 0.5 - self.marker_size/2)*self.rect_size, self.y0 + (y + 0.5 - self.marker_size/2)*self.rect_size, self.rect_size * self.marker_size, self.rect_size * self.marker_size, Qt.red)
                # painter.fillRect(self.x0 + x*self.rect_size, self.y0 + y*self.rect_size, self.rect_size, self.rect_size, Qt.black if array[y, x] == -1 else Qt.white)
        else:
            print("Surface to small")


class NetworkSurfaceWidget(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.helper = GridHelper(self.width, self.height, QColor(255, 255, 255, 255), False)
        self.values = numpy.empty((1, 1))
        self.pos = (-1, -1)

    def update_values(self, values, pos = (-1, -1)):
        self.values = values
        self.pos = pos

    def paintEvent(self, event : QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        self.helper.draw(painter, event.rect(), self.values, self.pos)
        painter.end()
        
    def save(self):
        now = datetime.now()
        x, y = self.size().width(), self.size().height()
        x = x - (x % self.values.shape[1])
        y = y - (y % self.values.shape[0])
        pixmap = QPixmap(x, y)
        self.helper.resize(QSize(x, y))
        painter = QPainter()
        painter.begin(pixmap)
        self.helper.draw(painter, QRect(), self.values, self.pos)
        painter.end()
        print(pixmap.save("../img/" + now.strftime("%d%m%Y_%H%M%S" + ".png"), "png"))
        self.helper.resize(self.size())

    def resizeEvent(self, event):
        self.helper.resize(event.size())
        self.update()

class NetworkWidget(QWidget):

    update_save = pyqtSignal()

    set_play = pyqtSignal(bool)

    def __init__(self, parent = None):
        QGLWidget.__init__(self, parent)
        self.network = HopfieldNetwork(random=True)
        self.network.randomise_values()
        self.network.order = 2

        self.mnist_handler = MNISTHandler()
        self.max_images = 5
        self.images = self.mnist_handler.get_pictures_by_label({x: self.max_images for x in range(10)})
        self.image_indizes = [0 for _ in range(10)]

        self.surface = NetworkSurfaceWidget(self)
        self.surface.helper.show_marker = True
        self.surface.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.surface.resize(self.size())
        self.update()
        

        self.timer = QTimer()
        self.timer.timeout.connect(self.go)

        self.noise_value = 0
        self.draw_mode = 0
        self.inpainting = False

        self.sym_action = None
        self.diag_action = None
        self.sync_action = None

    def go(self):
        self.network.iterate()
        self.update()

    def toggle_running(self, a : bool):
        if a and not self.timer.isActive():
            self.timer.start(40)
        elif not a and self.timer.isActive():
            self.timer.stop()
        self.update()

    def toggle_marker(self, a : bool):
        self.surface.helper.show_marker = a
        self.update()

    def random(self):
        self.network.randomise_weights()
        self.update_save_states()
        self.update()

    def clear(self):
        self.network.clear_values()
        self.update()

    def save(self):
        if self.network.saved_states.shape[0] < 10:
            self.network.save_state()
            self.network.calculate_weights()
            self.update_save_states()
        self.update()

    def make_image(self):
        self.surface.save()

    def update_save_states(self):
        self.update_save.emit()
        self.update()

    def reset(self):
        self.network.clear_weights_and_savedstates()
        self.update_save_states()
        self.update()

    def draw(self, a : bool):
        if a:
            self.draw_mode = 1
        self.update()

    def toggle_inpaint(self, a : bool):
        self.inpainting = a
        self.update()

    def erase(self, a : bool):
        if a:
            self.draw_mode = -1
        self.update()

    def noise(self):
        self.network.noise_values(self.noise_value)
        self.update()

    def step(self):
        s = self.network.step_size
        self.network.step_size = 1
        self.network.iterate()
        self.network.step_size = s
        self.update()

    def set_speed(self, speed : int):
        self.network.step_size = speed
        self.update()

    def set_noise_value(self, value : int):
        self.noise_value = value / 100
        self.update()

    def swap_order(self, index : int):
        self.network.order = index
        self.update()

    def show_save_state(self, idx : int):
        self.toggle_running(False)
        self.network.values = self.network.saved_states[idx].copy()
        self.set_play.emit(False)
        self.update()

    def delete_save(self, idx : int):
        self.network.remove_saved_state(idx)
        self.update_save_states()
        self.update()

    def load_number(self):
        number = int(self.sender().objectName().split('_')[2])
        self.network.load_image(self.images[number][self.image_indizes[number]])
        self.image_indizes[number] = (self.image_indizes[number] + 1) % self.max_images
        self.update()

    def toggle_symmetric(self, a : bool):
        self.network.weights.symetric = not self.network.weights.symetric
        self.update()

    def toggle_diagonal(self, a : bool):
        self.network.weights.zero_diagonal = not self.network.weights.zero_diagonal
        self.update()

    def toggle_synchronous(self, a : bool):
        self.network.set_sync(a)
        self.update()

    def load_example(self):
        idx = int(self.sender().objectName().split('_')[1])
        load_example(self.network, idx)
        self.sym_action.setChecked(self.network.weights.symetric)
        self.diag_action.setChecked(self.network.weights.zero_diagonal)
        self.sync_action.setChecked(self.network.sync)
        self.update()

    def mouseMoveEvent(self, event : QMoveEvent):
        x, y = event.pos().x(), event.pos().y()
        if self.surface.helper.x0 < x < self.surface.helper.xl:
            if self.surface.helper.y0 < y < self.surface.helper.yl:
                self.draw_to_network((x, y))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.surface.resize(self.size())
        self.surface.update()

    def update(self):
        self.surface.update_values(self.network.values, (self.network.current_x, self.network.current_y))
        self.surface.update()
        return super().update()

    def mousePressEvent(self, event : QMouseEvent):
        x, y = event.pos().x(), event.pos().y()
        if self.surface.helper.x0 < x < self.surface.helper.xl:
            if self.surface.helper.y0 < y < self.surface.helper.yl:
                self.draw_to_network((x, y))

    def draw_to_network(self, pos : tuple):
        x, y = pos
        if self.inpainting:
            self.network.values[int((y - self.surface.helper.y0) / self.surface.helper.rect_size)] = self.draw_mode
        else:
            self.network.values[(int((y - self.surface.helper.y0) / self.surface.helper.rect_size), int((x - self.surface.helper.x0) / self.surface.helper.rect_size))] = self.draw_mode
        self.update()