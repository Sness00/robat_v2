import os
import scipy.signal as signal
import numpy as np
from matplotlib import pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def sonar(signals, discarded_samples, fs, C_AIR=343):
    envelopes = np.abs(signal.hilbert(signals, axis=0))
    mean_envelope = np.mean(envelopes, axis=1)

    idxs, _ = signal.find_peaks(mean_envelope, prominence=1)
    try:
        emission_peak = idxs[0]    

        peaks = []
        enough = True
        for i in np.arange(envelopes.shape[1]):
            idxs, _ = signal.find_peaks(envelopes[emission_peak + discarded_samples:, i], prominence=6)
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
            return min_dist, emission_peak
    
        else:
            return 0, None
    except IndexError:
        t_plot = np.linspace(0, signals.shape[0]/fs, signals.shape[0])
        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
        plt.suptitle('Recorded Audio')
        for i in range(signals.shape[1]//2):
            for j in range(2):
                ax[i, j].plot(t_plot, signals[:, 2*i+j])
                ax[i, j].set_title('Channel %d' % (2*i+j+1))
                ax[i, j].minorticks_on()
                ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                ax[i, j].grid()
        plt.tight_layout()
        plt.show()
        plt.savefig('/logs/cross-corr.png')
        print('\nEmission peak not identified')        
        return 0, None
        