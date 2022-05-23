import csv
import sys
import math

from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QMainWindow, QWidget, QLineEdit
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox
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


class Window(QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.data = None

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        # Connections signal/slots
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionQuit.triggered.connect(self.close)
        self.ui.actionClose.triggered.connect(self.clear_chart)
        self.ui.actionPlot_Toolbar.triggered.connect(self.toggle_toolbar_visible)
        self.ui.pushButtonRecalculate.clicked.connect(self.recalculate)
        self.ui.toolButtonDataFile.clicked.connect(self.open_file)
        self.ui.actionAbout_FreeFFT.triggered.connect(self.about)
        self.ui.pushButtonReload.clicked.connect(self.reload_file)
        self.ui.checkBoxHideLowMagData.clicked.connect(self.checkBoxHideLowMagData_clicked)

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

    def load_csv(self, file_path=None):
        self.ui.labelBusy.setText('<h1>Reading CSV Data... (2/5)</h1>')
        self.repaint()
        
        if file_path:
            self.data = self.read_csv(file_path)
        
        if self.ui.checkBoxTimeSlice.isChecked():
            mask_time_slice_min = self.data[:,0] >= self.ui.doubleSpinBoxStartTime.value()
            self.data = self.data[mask_time_slice_min]
            mask_time_slice_max = self.data[:,0] <= self.ui.doubleSpinBoxEndTime.value()
            self.data = self.data[mask_time_slice_max]

        self.sample_size = len(self.data)
        self.time_elapsed = self.data[-1, 0] - self.data[0, 0]
        self.sampling_rate = self.sample_size / self.time_elapsed
        self.ui.lineEditSampleSize.setText(str(self.sample_size))

        self.plot()

    def plot(self, file_path=None):
        self.window_function()        

        self.rms_value = self.rms(self.data[:,1])
        self.peak_value = self.rms_value * math.sqrt(2)
        self.peak_peak_value = 2 * self.peak_value
        self.ui.lineEditRMS.setText(str(self.rms_value))
        self.ui.lineEditPeak.setText(str(self.peak_value))
        self.ui.lineEditPkPk.setText(str(self.peak_peak_value))

        # Plotted raw data (random sample of very large data sets)
        if self.ui.checkBoxLimitPlottedPoints.isChecked() and self.sample_size > self.ui.spinBoxMaxPlottedPoints.value():
            plot_row = np.random.randint(0, len(self.data), size=self.ui.spinBoxMaxPlottedPoints.value())
            plot_row.sort()
            self.plot_data = self.data[plot_row,:]
            self.plot_win = self.win[plot_row]
            self.plot_windowed_data = self.windowed_data[plot_row]
        else:
            self.plot_data = self.data
            self.plot_win = self.win
            self.plot_windowed_data = self.windowed_data

        # Calculate FFT
        self.ui.labelBusy.setText('<h1>Calculating FFT Data... (3/5)</h1>')
        self.repaint()
        freq, amplitude = fast_fourier_transform.make_fft(self.data[:, 0], self.windowed_data, self.sampling_rate)  
        self.fft_data = np.empty((len(freq), 2))
        self.fft_data[:, 0] = freq
        self.fft_data[:, 1] = amplitude

        self.max_amplitude = max(amplitude)
        self.min_amplitude = min(amplitude)

        # Eliminating FFT values below threashold
        if self.ui.checkBoxHideLowMagData.isChecked():
            mask_min_fft_val = self.fft_data[:, 1] > self.ui.doubleSpinBoxHideLowMagData.value() * self.max_amplitude
            self.plot_fft_data = self.fft_data[mask_min_fft_val]
        else:
            self.plot_fft_data = self.fft_data

        # Setting maximum frequency for plot
        if self.ui.checkBoxMaxPlotFreq.isChecked():
            mask_max_freq = self.plot_fft_data[:,0] <= self.ui.doubleSpinBoxMaxPlotFreq.value()
            self.plot_fft_data = self.plot_fft_data[mask_max_freq]

        if self.ui.checkBoxPlotFreqRes.isChecked():
            self.bins, self.amps = self.plot_freq_resolution(self.fft_data)
            self.binned_fft_data = np.empty((len(self.bins), 2))
            self.binned_fft_data[:,0] = self.bins
            self.binned_fft_data[:,1] = self.amps

            if self.ui.checkBoxHideLowMagData.isChecked():
                mask_min_bin_fft_val = self.binned_fft_data[:,1] > self.ui.doubleSpinBoxHideLowMagData.value() * self.max_amplitude
                self.plot_binned_fft_data = self.binned_fft_data[mask_min_bin_fft_val]
            else:
                self.plot_binned_fft_data = self.binned_fft_data

        label_sample_rate = np.round(self.sampling_rate, 2)
        self.ui.lineEditSamplingRate.setText(str(label_sample_rate))
        self.ui.lineEditNyquist.setText(str(np.round(label_sample_rate/2, 2)))

        # instead of ax.hold(False)
        self.figure.clear()

        # create an axis
        ax1 = self.figure.add_subplot(121)
        ax1twin = ax1.twinx()
        ax2 = self.figure.add_subplot(122)

        # Plot data
        self.ui.labelBusy.setText('<h1>Creating Raw Data Plot... (4/5)</h1>')
        self.repaint()
        ax1.plot(self.plot_data[:, 0], self.plot_data[:, 1], color='red')
        ax1twin.plot(self.plot_data[:,0], self.plot_win, color='green')
        ax1.plot(self.plot_data[:,0], self.plot_windowed_data, color='blue')
        self.ui.labelBusy.setText('<h1>Creating FFT Plot... (5/5)</h1>')
        self.repaint()

        if self.ui.checkBoxPlotFreqRes.isChecked():
            if self.ui.radioButtonChartContinuous.isChecked():
                ax2.plot(self.plot_binned_fft_data[:,0], self.plot_binned_fft_data[:,1], color="blue", linewidth=1)
            else:
                ax2.bar(self.plot_binned_fft_data[:,0], self.plot_binned_fft_data[:,1], color="blue", linewidth=0, width=3, align='center')
        else:
            if self.ui.radioButtonChartContinuous.isChecked():
                ax2.plot(self.plot_fft_data[:,0], self.plot_fft_data[:,1], color="blue", linewidth=1)
            else:
                ax2.bar(self.plot_fft_data[:,0], self.plot_fft_data[:,1], color="blue", linewidth=0, width=3, align='center')

        # plot text
        ax1.set_title('Raw Data')
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Amplitude')
        ax1twin.grid()

        ax2.set_title('FFT Data')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_xlim(left=0)


        # for point in self.fft_data:
        #     if point[1] >= self.max_amplitude * 0.1:
        #         ax2.text(point[0], point[1], str(point[1].round(2)) + " @ " + str(point[0].round(2)) + 'Hz')

        self.ui.busyWidget.hide()
        self.ui.chartWidget.show()
        self.toolbar.configure_subplots()._tight_layout()
        self.toolbar.configure_subplots().accept()
        self.canvas.draw()

    def open_file(self, file=None):
        if file:
            file_path = file
        else:
            file_path = QFileDialog.getOpenFileName(self, "Open File", __file__, "CSV files (*.csv)")[0]
            if file_path == '':
                return

        self.ui.labelBusy.setText('<h1>Opening File... (1/5)</h1>')
        self.ui.chartWidget.hide()
        self.ui.busyWidget.show()
        self.repaint()
        self.ui.lineEditDataFile.setText(file_path)
        self.repaint()
        self.load_csv(file_path)
        self.toolbar.show()
        self.ui.actionPlot_Toolbar.setChecked(True)


    def reload_file(self):
        file_path = self.ui.lineEditDataFile.text()
        if file_path == '':
            self.open_file()
            return
        else:
            self.open_file(file_path)


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
        self.ui.lineEditRMS.setText('')
        self.ui.lineEditPeak.setText('')
        self.ui.lineEditPkPk.setText('')

    def toggle_toolbar_visible(self):
        if self.ui.actionPlot_Toolbar.isChecked():
            self.toolbar.show()
        else:
            self.toolbar.hide()

    
    def rms(self, values):
        ms = np.sum(values**2) / len(values)
        return math.sqrt(ms)


    def window_function(self):
        if self.ui.radioButtonBlackman.isChecked():
            self.win = np.blackman(self.sample_size)
            self.windowed_data = self.data[:,1] * self.win
            return
        elif self.ui.radioButtonHanning.isChecked():
            self.win = np.hanning(self.sample_size)
            self.windowed_data = self.data[:,1] * self.win
            return
        elif self.ui.radioButtonHamming.isChecked():
            self.win = np.hamming(self.sample_size)
            self.windowed_data = self.data[:,1] * self.win
        elif self.ui.radioButtonKaiser.isChecked():
            self.win = np.kaiser(self.sample_size, float(self.ui.lineEditKaiserBeta.text()))
            self.windowed_data = self.data[:,1] * self.win        
        else:
            self.windowed_data = self.data
            self.win = np.full(len(self.data), 1)
            self.windowed_data = self.data[:,1] * self.win        


    def recalculate(self):
        if self.data is None:
            return
        self.ui.chartWidget.hide()
        self.ui.busyWidget.show()
        self.plot()


    def about(self):
        self.about_win = QMessageBox(self)
        self.about_win.setWindowTitle("About FreeFFT")
        self.about_win.setText("sdlkfjasdsjasa")
        self.about_win.show()

    def plot_freq_resolution(self, fft_data):
        step_size = self.ui.doubleSpinBoxPlotFreqResolution.value()
        if step_size <= 0:
            step_size = 1
            self.ui.doubleSpinBoxPlotFreqResolution.setValue(1.0)
        
        cur_bin_index = 0
        new_bins = [0]
        new_amps = [0]        
        for point in self.fft_data:
            if point[0] <= (cur_bin_index + 1) * step_size:
                new_amps[len(new_amps) - 1] += point[1]
            else:
                cur_bin_index += 1
                while point[0] > (cur_bin_index + 1) * step_size:
                    cur_bin_index += 1
                    new_amps.append(0)
                    new_bins.append((cur_bin_index + 1) * step_size)
                new_amps.append(point[1])
                new_bins.append((cur_bin_index + 1) * step_size)
        return new_bins, new_amps

    def checkBoxHideLowMagData_clicked(self):
        if self.ui.checkBoxHideLowMagData.isChecked():
            self.ui.radioButtonChartContinuous.setChecked(False)
            self.ui.radioButtonHistogram.setChecked(True)
            self.ui.radioButtonChartContinuous.setEnabled(False)
            self.ui.radioButtonHistogram.setEnabled(False)
        else:
            self.ui.radioButtonChartContinuous.setEnabled(True)
            self.ui.radioButtonHistogram.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.showMaximized()
    main.show()

    sys.exit(app.exec_())