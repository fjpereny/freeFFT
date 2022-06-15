import sys
import math

import PyQt5
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QMainWindow, QWidget, QLineEdit
from PyQt5.QtWidgets import QFileDialog, QDialog, QMessageBox, QLabel, QSplitter, QShortcut, QFrame
from PyQt5.Qt import QPen, QCursor, QMouseEvent
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt

from pyqtgraph import PlotWidget, plot, GraphicsWidget, PlotItem, GridItem, BarGraphItem, ViewBox, GraphicsLayoutWidget
import pyqtgraph as pg

import numpy as np
from numpy import array, random

from scipy import signal

import pandas as pd

from freefft import fast_fourier_transform
from freefft.statusbar_Vline import Statusbar_VLine
from freefft import themes
from freefft.ui_main import Ui_MainWindow
from freefft import ui_about_win

class Window(QMainWindow):
    def __init__(self, file=None):
        super(Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.about_win = None

        self.auto_calculate_enabled = True
        self.auto_resize_plot = True
    
        self.data = None
        self.data_plot = None
        self.filter_plot = None
        self.raw_plot = None
        self.fft_plot = None
        self.sync_plot = None

        # Statusbar setup
        self.mouse_mode_label = QLabel("Rect Select")
        self.auto_calculate_label = QLabel("Auto Calculate On")
        self.auto_resize_plot_label = QLabel("Auto Resize  On")
        self.file_label = QLabel("File: N/A")

        self.ui.statusbar.addPermanentWidget(Statusbar_VLine())
        self.ui.statusbar.addPermanentWidget(self.mouse_mode_label)
        self.ui.statusbar.addPermanentWidget(Statusbar_VLine())
        self.ui.statusbar.addPermanentWidget(self.auto_calculate_label)
        self.ui.statusbar.addPermanentWidget(Statusbar_VLine())
        self.ui.statusbar.addPermanentWidget(self.auto_resize_plot_label)
        self.ui.statusbar.addPermanentWidget(Statusbar_VLine())
        self.ui.statusbar.addPermanentWidget(self.file_label)
        self.ui.statusbar.addPermanentWidget(Statusbar_VLine())

        # Connections signal/slots
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionQuit.triggered.connect(self.close)
        self.ui.actionClose.triggered.connect(self.clear_chart)
        self.ui.actionRefresh_Data.triggered.connect(self.recalculate)
        self.ui.actionAbout_FreeFFT.triggered.connect(self.about)
        self.ui.actionAuto_Recalculate.triggered.connect(self.auto_recalculate_changed)
        self.ui.actionAuto_Resize_Plot.triggered.connect(self.auto_resize_plot_changed)

        self.ui.pushButtonReload.clicked.connect(self.reload_data)
        self.ui.pushButtonRecalculate.clicked.connect(self.recalculate)
        self.ui.toolButtonDataFile.clicked.connect(self.open_file)
        self.ui.pushButtonAddSignal.clicked.connect(self.add_sinusoid)
        # self.ui.pushButtonCalcConditionMonitoring.clicked.connect(self.calc_condition_monitoring)
        
        self.ui.spinBoxMinPower2.valueChanged.connect(self.power_2_preview)
        
        self.ui.comboBoxWindowOption.currentTextChanged.connect(self.window_option_changed)
        self.ui.comboBoxWindowOption.currentTextChanged.connect(self.check_auto_recalculate)
        self.ui.doubleSpinBoxKaiserBeta.valueChanged.connect(self.check_auto_recalculate)

        self.ui.checkBoxPadZeros.clicked.connect(self.check_auto_recalculate)
        self.ui.radioButtonMinPower2.clicked.connect(self.check_auto_recalculate)
        self.ui.radioButtonNearestPower2.clicked.connect(self.check_auto_recalculate)
        self.ui.spinBoxMinPower2.valueChanged.connect(self.check_auto_recalculate)
        self.ui.doubleSpinBoxMachineSpeed.valueChanged.connect(self.check_auto_recalculate)
        self.ui.doubleSpinBoxMaxSpeedVariation.valueChanged.connect(self.check_auto_recalculate)

        self.ui.checkBoxShowRawData.clicked.connect(self.replot_all)
        self.ui.checkBoxShowWindowedData.clicked.connect(self.replot_all)
        self.ui.checkBoxShowWindowFunction.clicked.connect(self.replot_all)
        self.ui.checkBoxShowZeroPadding.clicked.connect(self.replot_all)
        self.ui.checkBoxSynchSearch.clicked.connect(self.check_auto_recalculate)

        self.ui.radioButtonMachineSpeedHz.clicked.connect(self.check_auto_recalculate)
        self.ui.radioButtonMachineSpeedRPM.clicked.connect(self.check_auto_recalculate)

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

        theme_pallete = themes.ThemePalette()
        self.setPalette(theme_pallete)


    def window_option_changed(self):
        if self.ui.comboBoxWindowOption.currentText() == 'Kaiser-Bessel':
            self.ui.doubleSpinBoxKaiserBeta.setEnabled(True)
        else:
            self.ui.doubleSpinBoxKaiserBeta.setDisabled(True)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F5:
            self.recalculate()

        if event.key() == Qt.Key_Space:
            if self.data is None:
                pass
            else:
                self.auto_range_all()

        if event.key() == Qt.Key_Control:
            self.dataPlotViewBox.setMouseMode(3)
            self.fftPlotViewBox.setMouseMode(3)

            if self.dataPlot.isUnderMouse() or self.fftPlot.isUnderMouse():
                self.mouse_mode_label.setText("Pan  Mode")
                self.setCursor(Qt.OpenHandCursor)


    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.dataPlotViewBox.setMouseMode(1)
            self.fftPlotViewBox.setMouseMode(1)

            self.mouse_mode_label.setText("Rect Mode")
            self.setCursor(Qt.ArrowCursor)


    def auto_range_all(self):
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
        self.max_amplitude = max(self.fft_data[:,1])
        self.min_amplitude = min(self.fft_data[:,1])
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

        if self.ui.checkBoxSynchSearch.isChecked():
            self.synchronous_search_and_plot()


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
        
        self.replot_all()


    def read_csv(self, file_path, min_time=None, max_time=None):
        try:
            chunks = pd.read_csv(file_path, chunksize=1000)
            data = pd.concat(chunks)
            self.file_label.setText(file_path)
            return data.to_numpy()
        except TypeError as err:
            print(err)
            self.ui.labelBusy.setText('<h1>File Read Error: Unable to parse CSV file.</h1>\n<h2>Please ensure the selected file is in the correct format.</h2>')
            self.file_label.setText('File: Error')    
        except pd.errors.ParserError as err:
            print(err)
            self.ui.labelBusy.setText('<h1>File Read Error: Unable to parse CSV file.</h1>\n<h2>Please ensure the selected file is in the correct format.</h2>')  
            self.file_label.setText('File: Error')      
    
    
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
        selected_window = self.ui.comboBoxWindowOption.currentText()

        if selected_window =='Bartlett':
            self.win = signal.windows.bartlett(len(self.data))
            self.windowed_data = self.data[:,1] * self.win
            return     
        
        elif selected_window == 'Blackman':
            self.win = signal.windows.blackman(len(self.data))
            self.windowed_data = self.data[:,1] * self.win
            return
        
        elif selected_window == 'Hann':
            self.win = signal.windows.hann(len(self.data))
            self.windowed_data = self.data[:,1] * self.win
            return
        
        elif selected_window == 'Hamming':
            self.win = signal.windows.hamming(len(self.data))
            self.windowed_data = self.data[:,1] * self.win
        
        elif selected_window == 'Kaiser-Bessel':
            self.win = signal.windows.kaiser(len(self.data), self.ui.doubleSpinBoxKaiserBeta.value())
            self.windowed_data = self.data[:,1] * self.win        
        
        else:
            self.win = np.full(len(self.data), 1)
            self.windowed_data = self.data[:,1]        
        self.win_mean = sum(self.win) / self.sample_size


    def auto_resize_plot_changed(self):
        if self.ui.actionAuto_Resize_Plot.isChecked():
            self.auto_resize_plot = True
            self.auto_resize_plot_label.setText("Auto Resize  On")
            self.dataPlotViewBox.enableAutoRange()
            self.fftPlotViewBox.enableAutoRange()
            self.auto_range_all()
        else:
            self.auto_resize_plot = False
            self.auto_resize_plot_label.setText("Auto Resize Off")
            self.dataPlotViewBox.disableAutoRange()
            self.fftPlotViewBox.disableAutoRange()


    def auto_recalculate_changed(self):
        if self.ui.actionAuto_Recalculate.isChecked():
            self.auto_calculate_enabled = True
            self.auto_calculate_label.setText("Auto Calculate On")
            self.recalculate()
        else:
            self.auto_calculate_enabled = False
            self.auto_calculate_label.setText("Auto Calculate Off")


    def check_auto_recalculate(self):
        if self.auto_calculate_enabled:
            self.recalculate()
        else:
            return


    def recalculate(self):
        if self.data is None:
            return
        self.clear_chart()
        self.apply_window_function()


    def about(self):
        self.about_win = ui_about_win.Window()
        self.about_win.show()


    def power_2_preview(self):
        power = self.ui.spinBoxMinPower2.value()
        self.ui.lineEditPower2Preview.setText(str(2 ** power))

      
    def replot_all(self):
        self.clear_chart()
        if self.data is None:
            return
        self.plot()


    def add_sinusoid(self):
        A = self.ui.doubleSpinBoxAmplitude.value()
        f = self.ui.doubleSpinBoxFrequency.value()
        w = 2 * np.pi * f
        phi = self.ui.doubleSpinBoxPhase.value()

        time = self.data[:,0]
        x = A * np.sin(self.data[:,0] * w + phi)

        self.data[:,1] = self.data[:,1] + x
        self.check_auto_recalculate()


    def synchronous_search_and_plot(self):
        synch_speed = self.ui.doubleSpinBoxMachineSpeed.value()
        if self.ui.radioButtonMachineSpeedRPM.isChecked():
            synch_speed /= 60

        synch_harmonics = self.find_harmonics(synch_speed)

        found_harmonics_freq = []
        found_harmonic_amp = []

        for harmonic in synch_harmonics:
            nearest, index = self.find_nearest(self.fft_data[:,0], harmonic)
            nearest_max_index = self.find_nearest_max_harmonic(index, harmonic)
            found_harmonics_freq.append(self.fft_data[nearest_max_index,0])
            found_harmonic_amp.append(self.fft_data[nearest_max_index, 1])
        
        pen = pg.mkPen(color=(255, 0, 0), width=4)
        self.sync_plot = BarGraphItem(x=found_harmonics_freq, height=found_harmonic_amp, width=0, pen=pen)
        self.fftPlot.addItem(self.sync_plot)


    # Finds all possible harmonics based on nominal 1x synchronous
    def find_harmonics(self, synch_freq):
        i = 1
        synch_freqs = []
        while (i * synch_freq) <= self.nyquist_freq:
            synch_freqs.append(i * synch_freq)
            i += 1
        synch_freqs_array = np.array(synch_freqs)
        return synch_freqs_array


    # Finds the frequency closest to the nominal frequency harmonic
    def find_nearest(self, array, value):
        deltas = array - value
        deltas = np.abs(deltas)
        index = np.argmin(deltas)
        return array[index], index

    # Finds the maximum amplitude frequency within the bad of allowable slip frequencies
    def find_nearest_max_harmonic(self, index, harmonic):
        cur_max_amplitude = self.fft_data[index,1]
        cur_max_index = index
        max_freq = harmonic * (1 + self.ui.doubleSpinBoxMaxSpeedVariation.value()/100.0)
        min_freq = harmonic * (1 - self.ui.doubleSpinBoxMaxSpeedVariation.value()/100.0)

        # Searches left for local maximum amplitudes near nominal
        search_index = index
        while (search_index > 0):
            search_freq = self.fft_data[search_index,0]
            if search_freq < min_freq:
                break
            search_amplitude = self.fft_data[search_index,1]
            if search_amplitude > cur_max_amplitude:
                cur_max_amplitude = search_amplitude
                cur_max_index = search_index
                # print(cur_max_amplitude)
            search_index -= 1
            
        # Searches right for local maximum amplitudes near nominal
        search_index = index
        while (search_index < len(self.fft_data)):
            search_freq = self.fft_data[search_index,0]
            if search_freq > max_freq:
                break
            search_amplitude = self.fft_data[search_index,1]
            if search_amplitude > cur_max_amplitude:
                cur_max_amplitude = search_amplitude
                cur_max_index = search_index
                # print(cur_max_amplitude)
            search_index += 1

        return cur_max_index
        

def main():
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


if __name__ == '__main__':
    main()