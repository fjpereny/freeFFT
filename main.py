import sys
import math

import PyQt5
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QMainWindow, QWidget, QLineEdit
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox, QLabel, QSplitter, QShortcut
from PyQt5.Qt import QPen
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt

from pyqtgraph import PlotWidget, plot, GraphicsWidget, PlotItem, GridItem, BarGraphItem, ViewBox, GraphicsLayoutWidget
import pyqtgraph as pg

import numpy as np
from numpy import array, random

import pandas as pd

import fast_fourier_transform

# UI Imports
from ui_main import Ui_MainWindow
import ui_about_win

class Window(QMainWindow):
    def __init__(self, file=None):
        super(Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.about_win = None
    
        self.data = None
        self.data_plot = None
        self.filter_plot = None
        self.raw_plot = None
        self.fft_plot = None

        # Connections signal/slots
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionQuit.triggered.connect(self.close)
        self.ui.actionClose.triggered.connect(self.clear_chart)
        self.ui.actionAbout_FreeFFT.triggered.connect(self.about)

        self.ui.pushButtonReload.clicked.connect(self.reload_data)
        self.ui.pushButtonRecalculate.clicked.connect(self.recalculate)
        self.ui.toolButtonDataFile.clicked.connect(self.open_file)
        
        self.ui.spinBoxMinPower2.valueChanged.connect(self.power_2_preview)

        self.ui.checkBoxShowRawData.clicked.connect(self.replot_all)
        self.ui.checkBoxShowWindowedData.clicked.connect(self.replot_all)
        self.ui.checkBoxShowWindowFunction.clicked.connect(self.replot_all)
        self.ui.checkBoxShowZeroPadding.clicked.connect(self.replot_all)

        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('foreground', 'k')

        # Create the plots
        self.graphicsLayout = GraphicsLayoutWidget()
        self.dataPlot = PlotWidget(parent=self.graphicsLayout)
        self.fftPlot = PlotWidget(parent=self.graphicsLayout)
        self.dataPlotViewBox = self.dataPlot.plotItem.getViewBox()
        self.fftPlotViewBox = self.fftPlot.plotItem.getViewBox()

        # Configure the plots
        self.dataPlot.showGrid(x=True, y=True, alpha=1)
        self.fftPlot.showGrid(x=True, y=True, alpha=1)
        self.dataPlotViewBox.setMouseMode(1)
        self.fftPlotViewBox.setMouseMode(1)

        # set the layout
        chartLayout = QVBoxLayout()
        raw_data_label = QLabel(text="<h1>Raw Data</h1>")
        raw_data_label.setAlignment(Qt.AlignCenter)
        chartLayout.addWidget(raw_data_label)
        chartLayout.addWidget(self.dataPlot)
        self.ui.chartWidget.setLayout(chartLayout)

        fftLayout = QVBoxLayout()
        fft_data_label = QLabel(text="<h1>FFT Data</h1>")
        fft_data_label.setAlignment(Qt.AlignCenter)
        fftLayout.addWidget(fft_data_label)
        fftLayout.addWidget(self.fftPlot)
        self.ui.fftwidget.setLayout(fftLayout)
        self.ui.labelBusy.hide()

        self.splitter = QSplitter(parent=self)
        self.splitter.addWidget(self.ui.widgetLeftControls)
        self.splitter.addWidget(self.ui.chartWidget)
        self.splitter.addWidget(self.ui.fftwidget)
        self.splitter.addWidget(self.ui.labelBusy)
        self.setCentralWidget(self.splitter)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setStretchFactor(2, 4)
        self.splitter.setStretchFactor(3, 8)

        if file:
            self.open_file(file)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F5:
            self.recalculate()

        if event.key() == Qt.Key_Space:
            self.dataPlotViewBox.autoRange()
            self.fftPlotViewBox.autoRange()


    def load_csv(self, file_path):
        self.ui.labelBusy.setText('<h1>Reading CSV Data... (2/5)</h1>')
        self.ui.labelBusy.show()
        self.repaint()

        self.data = self.read_csv(file_path)
        
        if self.ui.checkBoxTimeSlice.isChecked():
            if self.ui.doubleSpinBoxEndTime.value() > self.ui.doubleSpinBoxStartTime.value():
                mask_time_slice_min = self.data[:,0] >= self.ui.doubleSpinBoxStartTime.value()
                self.data = self.data[mask_time_slice_min]
                mask_time_slice_max = self.data[:,0] <= self.ui.doubleSpinBoxEndTime.value()
                self.data = self.data[mask_time_slice_max]
            else:
                print('Warning: Invalid slice times. (Are min and max backwards?)')

        self.sample_size = len(self.data)
        self.time_elapsed = self.data[-1, 0] - self.data[0, 0]
        self.sampling_rate = self.sample_size / self.time_elapsed
        self.nyquist_freq = self.sampling_rate / 2

        self.apply_window_function()

    def apply_window_function(self):
        self.window_function()

        self.zero_fill_data = np.zeros((len(self.data), 2))
        self.zero_fill_data[:, 0] = self.data[:, 0]
        self.zero_fill_data[:, 1] = self.windowed_data

        # Zeros Padding
        if self.ui.checkBoxPadZeros.isChecked():
            if self.ui.radioButtonNearestPower2.isChecked():
                self.power_of_2 = fast_fourier_transform.find_nearest_power_2(len(self.data))
            else:
                min_points = 2 ** int(self.ui.spinBoxMinPower2.value())
                if min_points >= len(self.data):
                    self.power_of_2 = int(self.ui.spinBoxMinPower2.value())
                else:
                    self.power_of_2 = fast_fourier_transform.find_nearest_power_2(len(self.data))
        
            self.zero_fill_data = fast_fourier_transform.pad_zeros(self.zero_fill_data, self.power_of_2, self.sampling_rate)  
            self.zero_padded_size = len(self.zero_fill_data)
            self.ui.lineEditZeroPaddedSize.setText(str(self.zero_padded_size))   
        else:
            self.zero_padded_size = self.sample_size
            self.ui.lineEditZeroPaddedSize.setText('')

        self.make_fft()


    def make_fft(self):
        self.plot_data = self.zero_fill_data        
        self.ui.labelBusy.setText('<h1>Calculating FFT Data... (3/5)</h1>')
        self.repaint()
        freq, amplitude = fast_fourier_transform.make_fft(self.plot_data[:, 0], self.plot_data[:,1], self.sampling_rate, N=self.sample_size)  
        self.fft_data = np.empty((len(freq), 2))
        self.fft_data[:, 0] = freq
        self.fft_data[:, 1] = amplitude
        self.fft_data[:, 1] /= self.win_mean        
        self.max_amplitude = max(amplitude)
        self.min_amplitude = min(amplitude)
        self.plot()

    def plot(self, file_path=None):
        self.rms_value = np.round(self.rms(self.data[:,1]), 2)
        self.peak_value = np.round(self.rms_value * math.sqrt(2), 2)
        self.peak_peak_value = 2 * self.peak_value

        self.ui.lineEditSampleSize.setText(str(self.sample_size))
        self.ui.lineEditRMS.setText(str(self.rms_value))
        self.ui.lineEditPeak.setText(str(self.peak_value))
        self.ui.lineEditPkPk.setText(str(self.peak_peak_value))
        self.ui.lineEditSamplingRate.setText(str(np.round(self.sampling_rate, 2)))
        self.ui.lineEditNyquist.setText(str(np.round(self.sampling_rate/2, 2)))

        self.ui.labelBusy.setText('<h1>Plotting Raw Data... (4/5)</h1>')
        dataPen = pg.mkPen(color=(255, 0, 0), width=2)
        windowDataPen = pg.mkPen(color=(0, 150, 255), width=2)
        filterPen = pg.mkPen(color=(0, 255, 0), width=2)
        

        if self.ui.checkBoxShowRawData.isChecked():
            self.raw_plot = self.dataPlot.plot(self.data[:,0], self.data[:,1], pen=dataPen)
        
        if self.ui.checkBoxShowWindowedData.isChecked():
            if self.ui.checkBoxShowZeroPadding.isChecked():
                self.data_plot = self.dataPlot.plot(self.plot_data[:,0], self.plot_data[:,1], pen=windowDataPen)
            else:
                plot_data_no_zeros = np.resize(self.plot_data, (len(self.data), 2))
                self.data_plot = self.dataPlot.plot(plot_data_no_zeros[:,0], plot_data_no_zeros[:,1], pen=windowDataPen)
        
        if self.ui.checkBoxShowWindowFunction.isChecked():
            self.filter_plot = self.dataPlot.plot(self.data[:,0], self.win, pen=filterPen)

        self.ui.labelBusy.setText('<h1>Plotting FFT Data... (5/5)</h1>')
        fftPen = pg.mkPen(color=(0, 150, 255), width=2)
        self.fft_plot = BarGraphItem(x=self.fft_data[:,0], height=self.fft_data[:,1], width=0, pen=fftPen)
        self.fftPlot.addItem(self.fft_plot)
        
        self.ui.labelBusy.hide()
        self.ui.chartWidget.show()
        self.ui.fftwidget.show()


    def open_file(self, file=None):
        if file:
            file_path = file
        else:
            file_path = QFileDialog.getOpenFileName(self, "Open File", __file__, "CSV files (*.csv)")[0]
            if file_path == '':
                return

        self.ui.labelBusy.show()
        self.ui.labelBusy.setText('<h1>Opening File... (1/5)</h1>')
        self.ui.chartWidget.hide()
        self.ui.fftwidget.hide()
        self.ui.lineEditDataFile.setText(file_path)
        self.clear_chart()
        self.repaint()
        self.load_csv(file_path)


    def reload_data(self):
        file_path = self.ui.lineEditDataFile.text()
        if file_path == '':
            self.open_file()
        else:
            self.load_csv(file_path)


    def read_csv(self, file_path, min_time=None, max_time=None):
        try:
            chunks = pd.read_csv(file_path, chunksize=1000)
            data = pd.concat(chunks)
            return data.to_numpy()
        except TypeError as err:
            print(err)
            self.ui.labelBusy.setText('<h1>File Read Error: Unable to parse CSV file.</h1>\n<h2>Please ensure the selected file is in the correct format.</h2>')    
        except pd.errors.ParserError as err:
            print(err)
            self.ui.labelBusy.setText('<h1>File Read Error: Unable to parse CSV file.</h1>\n<h2>Please ensure the selected file is in the correct format.</h2>')    
    
    
    def clear_chart(self):
        self.dataPlot.clear()
        self.fftPlot.clear()
        self.ui.lineEditSampleSize.setText('')
        self.ui.lineEditSamplingRate.setText('')
        self.ui.lineEditNyquist.setText('')
        self.ui.lineEditRMS.setText('')
        self.ui.lineEditPeak.setText('')
        self.ui.lineEditPkPk.setText('')

    
    def rms(self, values):
        ms = np.sum(values**2) / len(values)
        return math.sqrt(ms)


    def window_function(self):
        if self.ui.radioButtonBlackman.isChecked():
            self.win = np.blackman(len(self.data))
            self.windowed_data = self.data[:,1] * self.win
            return
        elif self.ui.radioButtonHanning.isChecked():
            self.win = np.hanning(len(self.data))
            self.windowed_data = self.data[:,1] * self.win
            return
        elif self.ui.radioButtonHamming.isChecked():
            self.win = np.hamming(len(self.data))
            self.windowed_data = self.data[:,1] * self.win
        elif self.ui.radioButtonKaiser.isChecked():
            self.win = np.kaiser(len(self.data), self.ui.doubleSpinBoxKaiserBeta.value())
            self.windowed_data = self.data[:,1] * self.win        
        else:
            self.win = np.full(len(self.data), 1)
            self.windowed_data = self.data[:,1]        
        self.win_mean = sum(self.win) / self.sample_size


    def recalculate(self):
        if self.data is None:
            return
        self.clear_chart()
        self.apply_window_function()


    def about(self):
        self.about_win = ui_about_win.Window()
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

        # create an axis
        self.axWaterfall.plot(self.plot_fft_data[:, 0] * self.increment, np.ones(len(self.plot_fft_data)) * max(self.plot_data[:,0]) * self.increment, self.plot_fft_data[:, 1] * self.increment)
        self.axWaterfall.set_title('FFT Waterfall Plot')
        self.axWaterfall.set_xlabel('Frequency (Hz)')
        self.axWaterfall.set_ylabel('Time (sec)')
        self.axWaterfall.set_zlabel('Amplitude')
        self.axWaterfall.grid()
        
        self.canvas.draw()
        self.repaint()


    def power_2_preview(self):
        power = self.ui.spinBoxMinPower2.value()
        self.ui.lineEditPower2Preview.setText(str(2 ** power))

      
    def replot_all(self):
        self.clear_chart()
        if self.data is None:
            return
        self.plot()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Allows opening file directly from terminal
    if len(sys.argv) > 1:
        file = sys.argv[1]
        main = Window(file)
    else:
        main = Window()
    main.showMaximized()
    main.show()

    sys.exit(app.exec_())