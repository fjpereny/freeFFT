import sys

from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QMainWindow, QWidget, QLineEdit
from PyQt5.QtWidgets import QFileDialog

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import numpy as np
from numpy import array

import fast_fourier_transform
from ui_main import Ui_MainWindow
import settings

class DataPoint(object):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Window(QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.widget = QWidget(self)
        # self.setCentralWidget(self.widget)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        # Just connections
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionQuit.triggered.connect(self.close)
        self.ui.actionClose.triggered.connect(self.clear_chart)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.ui.widget.setLayout(layout)

    def plot(self, file_path=None):
        min_time = settings.RAW_MIN_TIME
        max_time = settings.RAW_MAX_TIME

        self.data = fast_fourier_transform.read_csv(file_path, min_time, max_time)
        data_slice = self.data[:]

        # Calculate FFT
        freq, amplitude, sample_rate = fast_fourier_transform.make_fft(data_slice[:, 0], data_slice[:, 1])

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)

        # plot data
        ax1.plot(data_slice[:, 0], data_slice[:, 1], color='red')
        ax2.stem(freq, amplitude, linefmt='b', markerfmt=" ", basefmt="-b")
        ax2.set_xlim([0, sample_rate//2])

        # plot text
        ax1.set_title('Raw Data')
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Amplitude')

        ax2.set_title('FFT Data')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')

        # refresh canvas
        self.canvas.draw()

    def open_file(self):
        file_path = QFileDialog.getOpenFileName(self, "Open File", "/home", "CSV files (*.csv)")
        if file_path:
            self.plot(file_path[0])
            self.toolbar.show()

    def clear_chart(self):
        self.figure.clear()
        self.toolbar.hide()
        self.canvas.draw()
        self.data = None

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.showMaximized()
    main.show()

    sys.exit(app.exec_())