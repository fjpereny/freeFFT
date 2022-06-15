import sys

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication

from freefft.ui_about import Ui_AboutWindow


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.ui = Ui_AboutWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.close)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())