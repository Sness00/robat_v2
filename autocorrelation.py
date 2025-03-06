import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def pow_two_pad_and_window(vec, fs, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.5)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(padded_windowed_vec))
        plt.figure()
        plt.plot(t, padded_windowed_vec)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)

fs = 192000

dur = 3e-3
hi_freq = 60e3
low_freq = 20e3

t_tone = np.linspace(0, dur, int(fs*dur))
chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)

sig = pow_two_pad_and_window(chirp, fs)

delayed_sig = np.roll(sig, 150)

autocorr = signal.correlate(sig, delayed_sig, 'full', method='fft')

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(sig)
plt.subplot(3, 1, 2)
plt.plot(delayed_sig)
plt.subplot(3, 1, 3)
plt.plot(autocorr)
plt.show()