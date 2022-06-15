from random import sample
import numpy as np
from numpy.fft import fft


def make_fft(time_vals, amplitude_vals, sampling_rate, N=None, n=None):
    if len(amplitude_vals) != len(time_vals):
        print("Error: Amplitude and time valudes must have the same length.")
        return None
    
    X = fft(amplitude_vals, n=n)
    if N == None:
        N = len(X)
    else:
        N = N

    # sampling_rate = N / (max(time_vals) - min(time_vals))     
    T = len(X) / sampling_rate

    unique_points = np.ceil((N+1)/2)
    n = np.arange(unique_points)
    freq = n/T
    
    # Remove all values greater than Nyquist frequency
    X = X[:len(freq)]
    amplitudes = np.abs(X)/len(freq)
    amplitudes[0] /= 2 # DC component should not be multiplied by 2

    return [freq, amplitudes]


def find_nearest_power_2(num_of_ponts):
    x = 0
    while 2**x < num_of_ponts:
        x += 1
    return x


def pad_zeros(data, power_of_2, sampling_rate):
    x = 2**power_of_2
    n = x - len(data)
    
    if n == 0:
        return data
    
    max_time = data[-1,0]
    
    i = 1
    timesArr = np.zeros((n,2))
    while i <= n:
        timesArr[i-1, 0] = (max_time + i / sampling_rate)
        i += 1
    
    new_data = np.row_stack((data, timesArr))
    return new_data