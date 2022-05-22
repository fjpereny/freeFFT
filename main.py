import csv
import sys
import math

from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QMainWindow, QWidget, QLineEdit
from PyQt5.QtWidgets import QFileDialog
from PyQt5.Qt import QMovie
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import numpy as np
from numpy import array, random

import seaborn
seaborn.set()

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
        self.ui.actionPlot_Toolbar.triggered.connect(self.toggle_toolbar_visible)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.ui.chartWidget.setLayout(layout)

        # set loading move to label object
        # movie = QMovie('loading-blocks.gif')
        # movie.start()
        # self.ui.labelBusyGif.setMovie(movie)
        self.ui.busyWidget.hide()

    def plot(self, file_path=None):
        min_time = settings.RAW_MIN_TIME
        max_time = settings.RAW_MAX_TIME

        self.ui.labelBusy.setText('<h1>Reading CSV Data... (2/5)</h1>')
        self.repaint()
        self.data = self.read_csv(file_path, min_time, max_time)
        self.sample_size = len(self.data)
        self.time_elapsed = self.data[-1, 0] - self.data[0, 0]
        self.sampling_rate = self.sample_size / self.time_elapsed
        self.ui.lineEditSampleSize.setText(str(self.sample_size))

        self.rms_value = self.rms(self.data[:,1])
        self.peak_value = self.rms_value * math.sqrt(2)
        self.peak_peak_value = 2 * self.peak_value

        # Plotted raw data (random sample of very large data sets)
        if self.sample_size > settings.MAX_RAW_PLOT_COUNT:
            plot_row = np.random.randint(0, len(self.data), size=settings.MAX_RAW_PLOT_COUNT)
            plot_row.sort()
            self.plot_data = self.data[plot_row, :]
        else:
            self.plot_data = self.data

        # Calculate FFT
        self.ui.labelBusy.setText('<h1>Calculating FFT Data... (3/5)</h1>')
        self.repaint()
        freq, amplitude = fast_fourier_transform.make_fft(self.data[:, 0], self.data[:, 1], self.sampling_rate)  
        self.fft_data = np.empty((len(freq), 2))
        self.fft_data[:, 0] = freq
        self.fft_data[:, 1] = amplitude

        self.max_amplitude = max(amplitude)
        self.min_amplitude = min(amplitude)

        # Reducing number of plotted points for FFT
        mask_min_fft_val = self.fft_data[:, 1] > settings.MIN_FFT_PCT * self.max_amplitude
        self.plot_fft_data = self.fft_data[mask_min_fft_val]

        label_sample_rate = np.round(self.sampling_rate, 2)
        self.ui.lineEditSamplingRate.setText(str(label_sample_rate) + " Hz")
        self.ui.lineEditNyquist.setText(str(np.round(label_sample_rate/2, 2)) + "Hz")

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)

        # Plot data
        self.ui.labelBusy.setText('<h1>Creating Raw Data Plot... (4/5)</h1>')
        self.repaint()
        ax1.plot(self.plot_data[:, 0], self.plot_data[:, 1], color='red')
        self.ui.labelBusy.setText('<h1>Creating FFT Plot... (5/5)</h1>')
        self.repaint()
        ax2.bar(self.plot_fft_data[:, 0], self.plot_fft_data[:, 1], align='center', width=3, color="blue", linewidth=0)

        # plot text
        ax1.set_title('Raw Data')
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Amplitude')

        ax2.set_title('FFT Data')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_xlim(left=0)
        ax2.set_ylim(top=self.max_amplitude*1.2)

        self.ui.busyWidget.hide()
        self.ui.chartWidget.show()
        self.toolbar.configure_subplots()._tight_layout()
        self.toolbar.configure_subplots().accept()
        self.canvas.draw()

    def open_file(self):
        file_path = QFileDialog.getOpenFileName(self, "Open File", __file__, "CSV files (*.csv)")
        if file_path[0] == '':
            return
        self.ui.labelBusy.setText('<h1>Opening File... (1/5)</h1>')
        self.ui.chartWidget.hide()
        self.ui.busyWidget.show()
        self.repaint()
        self.ui.lineEditDataFile.setText(file_path[0])
        self.repaint()
        self.plot(file_path[0])
        self.toolbar.show()
        self.ui.actionPlot_Toolbar.setChecked(True)

    def read_csv(self, file_path, min_time=None, max_time=None):
        points = []
        with open(file_path) as file:
            reader = csv.reader(file)
            for row in reader:
                time = float(row[0])
                if min_time != None and time < min_time:
                    continue
                elif max_time != None and time > max_time:
                    return np.array(points)
                else:
                    amplitude = float(row[1])
                    points.append([time, amplitude])
        return np.array(points)

    def clear_chart(self):
        self.figure.clear()
        self.toolbar.hide()
        self.ui.actionPlot_Toolbar.setChecked(False)
        self.canvas.draw()
        self.data = None
        self.ui.lineEditSampleSize.setText('')
        self.ui.lineEditSamplingRate.setText('')
        self.ui.lineEditDataFile.setText('')
        self.ui.lineEditNyquist.setText('')

    def toggle_toolbar_visible(self):
        if self.ui.actionPlot_Toolbar.isChecked():
            self.toolbar.show()
        else:
            self.toolbar.hide()

    
    def rms(self, values):
        ms = np.sum(values**2) / len(values)
        print(ms)
        return math.sqrt(ms)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.showMaximized()
    main.show()

    sys.exit(app.exec_())