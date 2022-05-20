import csv
from random import sample
import numpy as np
from numpy.fft import fft


def read_csv(file_path):
    points = []
    with open(file_path) as file:
        reader = csv.reader(file)
        for row in reader:
            time = float(row[0])
            amplitude = float(row[1])
            points.append([time, amplitude])
    return np.array(points)


def read_csv(file_path, min_time, max_time):
    points = []
    with open(file_path) as file:
        reader = csv.reader(file)
        for row in reader:
            time = float(row[0])
            if time < min_time:
                continue
            elif time > max_time:
                return np.array(points)
            else:
                amplitude = float(row[1])
                points.append([time, amplitude])
    return np.array(points)


def make_fft(time_vals, amplitude_vals):
    if len(amplitude_vals) != len(time_vals):
        print("Error: Amplitude and time valudes must have the same length.")
        return None
    
    if min(time_vals) < 0:
        print("Error: Time values cannot be negative values.")
        return None
    
    X = fft(amplitude_vals)
    N = len(X)

    time_elapsed = max(time_vals) - min(time_vals)
    sample_rate = N / time_elapsed
    T = N/sample_rate

    n = np.arange(N)
    freq = n/T

    # Limit the maximum frequency value to the Nyquist frequnecy (1/2 sampling rate)
    freq = [x for x in freq if x <= (sample_rate//2)]
    
    # Remove all values greater than Nyquist frequency
    X = X[:len(freq)]
    amplitudes = np.abs(X)/N*2

    return [freq, amplitudes, sample_rate]
