import PyQt5

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtOpenGL


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *

from util.widgets.hopfield_main_window import HopfieldMainWindow

if __name__ == "__main__":
    import sys
    QApplication.setStyle('Fusion')
    app = QApplication(sys.argv)
    MainWindow = HopfieldMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())