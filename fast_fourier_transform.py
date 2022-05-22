from random import sample
import numpy as np
from numpy.fft import fft


def make_fft(time_vals, amplitude_vals, sampling_rate, n=None):
    if len(amplitude_vals) != len(time_vals):
        print("Error: Amplitude and time valudes must have the same length.")
        return None
    
    if min(time_vals) < 0:
        print("Error: Time values cannot be negative values.")
        return None
    
    X = fft(amplitude_vals, n=n)
    N = len(X)
    # sampling_rate = N / (max(time_vals) - min(time_vals))     
    T = N / sampling_rate

    unique_points = np.ceil((N+1)/2)
    n = np.arange(unique_points)
    freq = n/T
    
    # Remove all values greater than Nyquist frequency
    X = X[:len(freq)]
    amplitudes = np.abs(X)/len(freq)
    amplitudes[0] /= 2 # DC component should not be multiplied by 2

    return [freq, amplitudes]
