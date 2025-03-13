import scipy.signal as signal
import numpy as np

def sonar(signals, discarded_samples, fs, C_AIR=343):
    envelopes = np.abs(signal.hilbert(signals, axis=0))
    mean_envelope = np.sum(envelopes, axis=1)/envelopes.shape[1]

    idxs, _ = signal.find_peaks(mean_envelope, prominence=7)
    emission_peak = idxs[0]    

    peaks = []
    enough = True
    for i in np.arange(envelopes.shape[1]):
        idxs, _ = signal.find_peaks(envelopes[emission_peak + discarded_samples:, i], prominence=5)
        if idxs.any():
            peaks.append(idxs[0] + emission_peak + discarded_samples)
        else:
            enough = False
            break
            
    if enough:
        min_dist = np.inf
        for p in peaks:
            dist = (p - emission_peak)/fs*C_AIR/2 + 0.025
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    else:
        return 0
        