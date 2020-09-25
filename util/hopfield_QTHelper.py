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

from util.hopfield_util import HopfieldNetwork, MNISTHandler, load_example

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

    def save(self, dialog = True):
        file_type = "png"
        now = datetime.now()
        defaul_path = "./img/" + now.strftime("%d%m%Y_%H%M%S" + ".png")
        if dialog:
            try:
                f = QFileDialog.getSaveFileName(self.parent(), "FileDialog", defaul_path, "PNG (*.png);;BMP (*.bmp);;JPG (*.jpg)", "PNG (*.png)")
                path = f[0]
                if "bmp" in f[1]:
                    file_type = "bmp"
                elif "jpg" in f[1]:
                    file_type = "jpg"
            except Exception as e:
                print(e)
        else:
            path = defaul_path
        if len(path) == 0:
            return
        m = self.helper.show_marker
        self.helper.show_marker = False
        x, y = self.size().width(), self.size().height()
        x = x - (x % self.values.shape[1])
        y = y - (y % self.values.shape[0])
        s = min(x, y)
        pixmap = QPixmap(s, s)
        self.helper.resize(QSize(s, s))
        painter = QPainter()
        painter.begin(pixmap)
        self.helper.draw(painter, QRect(), self.values, self.pos)
        painter.end()
        pixmap.save(path, file_type)
        self.helper.resize(self.size())
        self.helper.show_marker = m

    def resizeEvent(self, event):
        self.helper.resize(event.size())
        self.update()

class NetworkWidget(QWidget):

    update_save = pyqtSignal()

    set_play = pyqtSignal(bool)

    update_step_label = pyqtSignal(str)

    def __init__(self, parent = None):
        QGLWidget.__init__(self, parent)
        self.steps = 0
        self.network = HopfieldNetwork(random=True)
        self.network.noise_values(1)
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

        self.noise_value = 0.3
        #1: white, -1: black
        self.draw_color = 1
        #0:point, 1:circle, 2:line
        self.draw_mode = 0

        self.sym_check = None
        self.diag_check = None
        self.async_check = None

    def go(self):
        self.network.iterate()
        self.steps += self.network.step_size
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
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def save(self):
        if self.network.saved_states.shape[0] < 10:
            self.network.save_state()
            self.network.calculate_weights()
            self.update_save_states()
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def save_image(self):
        self.surface.save(False)

    def save_image_as(self):
        self.surface.save(True)

    def update_save_states(self):
        self.update_save.emit()
        self.update()

    def reset(self):
        self.network.clear_weights_and_savedstates()
        self.update_save_states()
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def draw(self):
        self.draw_color = 1
        self.update()

    def erase(self):
        self.draw_color = -1
        self.update()

    def set_draw_color(self, a : bool):
        self.draw_color = -1 if a else 1

    def set_draw_mode(self, idx : int):
        self.draw_mode = idx

    def noise(self):
        self.network.noise_values(self.noise_value)
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def invert(self):
        self.network.values *= -1
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def step(self):
        s = self.network.step_size
        self.network.step_size = 1
        self.go()
        self.network.step_size = s

    def complete_iteration(self):
        s = self.network.step_size
        self.network.step_size = 1
        self.go()
        while len(self.network.remaining_indizes) > 0:
            self.go()
        
        self.network.step_size = s

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
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def delete_save(self, idx : int):
        self.network.remove_saved_state(idx)
        self.update_save_states()
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def load_number(self):
        number = int(self.sender().objectName().split('_')[2])
        self.network.load_image(self.images[number][self.image_indizes[number]])
        self.image_indizes[number] = (self.image_indizes[number] + 1) % self.max_images
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def toggle_symmetric(self, a : bool):
        self.network.weights.symetric = not self.network.weights.symetric
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def toggle_diagonal(self, a : bool):
        self.network.weights.zero_diagonal = not self.network.weights.zero_diagonal
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def toggle_asynchronous(self, a : bool):
        self.network.set_sync(not a)
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def load_example(self):
        idx = int(self.sender().objectName().split('_')[1])
        load_example(self.network, idx)
        self.sym_check.setChecked(self.network.weights.symetric)
        self.diag_check.setChecked(self.network.weights.zero_diagonal)
        self.async_check.setChecked(not self.network.sync)
        self.steps = 0
        self.network.remaining_indizes = []
        self.update()

    def update_steps(self):
        s = self.steps / (self.network.w * self.network.h)
        string = " : ".join([str(self.steps), f"{s:.3f}"])
        self.update_step_label.emit(string)

    def mouseMoveEvent(self, event : QMoveEvent):
        x, y = event.pos().x(), event.pos().y()
        if self.surface.helper.x0 < x < self.surface.helper.xl:
            if self.surface.helper.y0 < y < self.surface.helper.yl:
                self.draw_to_network((x, y))

    def mousePressEvent(self, event : QMouseEvent):
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
        self.update_steps()
        return super().update()

    def draw_to_network(self, pos : tuple):
        self.steps = 0
        self.network.remaining_indizes = []
        y, x = (int((pos[1] - self.surface.helper.y0) / self.surface.helper.rect_size), int((pos[0] - self.surface.helper.x0) / self.surface.helper.rect_size))
        if self.draw_mode == 0:
            self.network.values[y, x] = self.draw_color
        elif self.draw_mode == 1:
            self.network.values[max(0,y-2):y+3, max(0,x-2):x+3] = self.draw_color
        elif self.draw_mode == 2:
            self.network.values[y] = self.draw_color
        elif self.draw_mode == 3:
            self.network.values = numpy.ones(self.network.network_size) * self.draw_color
        self.update()