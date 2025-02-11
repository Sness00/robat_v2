import numpy as np
from scipy.signal import stft

def das_filter(y, fs, nch, d, bw, theta=np.linspace(-90, 90, 36), c=343):
    """
    Simple multiband Delay-and-Sum spatial filter implementation.
    Parameters:
    - y: mic array signals
    - fs: sampling rate
    - nch: number of mics in the array
    - d: mic spacing
    - bw: (low freq, high freq)
    - theta: angle vector
    - c: sound speed

    Returns: average pseudospectrum across the various bands
    """
    f_spec_axis, t_spec_axis, spectrum = stft(y, fs=fs, nperseg=64, noverlap=63, axis=0)
    bands = f_spec_axis[(f_spec_axis > bw[0]) & (f_spec_axis < bw[1])]
    bands = np.array([bands[0], bands[len(bands)//2], bands[-1]])

    a = np.zeros((nch, len(theta), len(bands)), dtype=complex)
    cov_est = np.zeros((nch, nch, len(bands)), dtype=complex)    
    for f_c in bands:
        a[:, :, bands == f_c] = np.expand_dims(np.exp(-1j * 2 * np.pi * f_c * d * np.sin(np.deg2rad(theta)) * np.linspace(0, nch-1, nch)[:, None] / c), 2)
        for ii in range(len(t_spec_axis)):
            spec = spectrum[f_spec_axis == f_c, :, ii].squeeze()
            cov_est[:, :, bands == f_c] += np.expand_dims(np.outer(spec, spec.conj().T), 2) / len(t_spec_axis)

    p = np.zeros((len(bands), len(theta)), dtype=complex)
    for ii in range(len(bands)):
        for jj in range(len(theta)):
            p[ii, jj] = a[:, jj, ii].conj().T @ cov_est[:, :, ii] @ a[:, jj, ii] / nch**2
    
    avg_pseudo_spec = np.sum(p, axis=0) / len(bands)
    return theta, np.abs(avg_pseudo_spec)