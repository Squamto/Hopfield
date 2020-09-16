import PyQt5

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtOpenGL

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *

from util.ui.main_window import Ui_MainWindow


class HopfieldMainWindow(QMainWindow):

    def __init__(self, parent = None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.networkWidget.sym_action = self.ui.action_symmetric
        self.ui.networkWidget.diag_action = self.ui.action_diagonal
        self.ui.networkWidget.sync_action = self.ui.action_synchronous




if __name__ == "__main__":
    import sys
    QApplication.setStyle('Fusion')
    app = QApplication(sys.argv)
    MainWindow = HopfieldMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())