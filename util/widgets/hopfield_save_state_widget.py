import PyQt5

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtOpenGL

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *

from util.ui.save_state import Ui_SaveState
from util.hopfield_QTHelper import NetworkWidget

from random import random

class SaveStateWidget(QFrame):

    show_request = pyqtSignal(int)
    delete_request = pyqtSignal(int)

    def __init__(self, parent : NetworkWidget):
        super().__init__(parent)
        self.ui = Ui_SaveState()
        self.ui.setupUi(self)
        self.ui.network_surface.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ui.network_surface.adjustSize()
        self.hide()

    def setObjectName(self, str):
        super().setObjectName(str)
        if str == "SaveState":
            return
        self.idx = int(str[6:]) - 1
        self.update_index()

    def update_index(self):
        self.ui.label.setText(str(self.idx))

    def update_state(self):
        sender = self.sender()
        if sender.network.saved_states.shape[0] <= self.idx:
            self.hide()
            return
        self.ui.network_surface.update_values(self.sender().network.saved_states[self.idx])
        self.ui.network_surface.update()
        self.show()

    def view(self):
        self.show_request.emit(self.idx)

    def clear(self):
        self.delete_request.emit(self.idx)