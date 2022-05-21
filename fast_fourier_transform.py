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

    n = np.arange(N/2)
    freq = n/T
    
    # Remove all values greater than Nyquist frequency
    X = X[:len(freq)]
    amplitudes = np.abs(X)/N*2

    return [freq, amplitudes]
