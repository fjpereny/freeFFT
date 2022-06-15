# FreeFFT
FreeFFT is a [free](https://www.gnu.org/philosophy/free-sw.html) and open source vibration analsys software package.

<img src="https://github.com/fjpereny/freeFFT/blob/main/img/FreeFFT%20GUI%201.png?raw=true" alt="FreeFFT GUI">

## Features
- Simple & Intuitive Interface
- Two Column CSV Data Import Compatibility
- Time Slicing of Imported Data
- Autodetected Sampling Frequency & Nyquist Frequency
- RMS, Peak, and Peak-Peak Calculation
- Fast Fourier Transform (FFT)
- Data Zero Padding
- Multitude of Windowing Functions
- Export of Resulting Plots and Calculated Data

## System Requirements
FreeFFT is compatible with Linux, FreeBSD, MacOS, and Windows systems that have Python 3 installed.  

### Python Requirements
FreeFFT requires Python 3 and the following packages:
 - PyQt5
 - pyqtgraph
 - numpy
 - scipy
 - pandas

### Installing
Free FFT can be installed via PIP with the following terminal command:
```
pip install freefft
```

## Installing & Running
Simply pull the repository or download and extract all files into the directory of your preference.  The application is started by running main.py.  Optionally, main.py takes an argument to directly open a data file.

Opening the application:
```
python -m freefft
````
Opening the application directly with data file:
```
python -m freefft data.csv
```

## License
This software is released uner [GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007](https://www.gnu.org/licenses/gpl-3.0.en.html).

Copyright © 2022 [Frank Pereny](https://github.com/fjpereny/)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
