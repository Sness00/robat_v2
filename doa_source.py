import os
import time
from datetime import datetime
import queue
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter

def get_soundcard_instream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')

def windower(a):
    window = signal.windows.tukey(len(a), alpha=0.2)
    if len(a.shape) > 1:
        window = np.reshape(window, (-1, 1))
    windowed_a = a * window
    return windowed_a

if __name__ == "__main__":

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    fs = 192000
    C_AIR = 343
    nch = 8

    first_sample = int(fs*0.1)
    last_sample = first_sample + int(3e-3*fs)
    sig, _ = sf.read('15k-95k.wav', start=first_sample, stop=last_sample)
    
    field_range = 50e-2
    discarded_samples = int(np.floor((field_range*2)/C_AIR*fs)) - 60
    processed_samples = 512

    hi_freq = 95e3
    low_freq = 15e3

    rec_dir = './recordings/'
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    audio_in_data = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_in_data.put(indata.copy())

    activate_mics()

    try:        
        now = datetime.now()
        filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S') + '.wav')
        with sf.SoundFile(filename, mode='x', samplerate=int(fs),
                            channels=8) as file:
            stream = sd.InputStream(samplerate=fs,
                                device=get_soundcard_instream(sd.query_devices()), 
                                channels=8, 
                                callback=callback,
                                latency='low',
                                blocksize=0 
                                )
            print('Stream started')
            with stream:
                while True:
                    file.write(audio_in_data.get())

    except KeyboardInterrupt:
        print('Stream closed')

        input_audio, _ = sf.read(filename)

        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
        plt.suptitle('Recorded Audio')
        for i in range(input_audio.shape[1]//2):
            for j in range(2):
                ax[i, j].plot(input_audio[:, 2*i+j])
                ax[i, j].set_title('Channel %d' % (2*i+j+1))
                ax[i, j].minorticks_on()
                ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                ax[i, j].grid()
        plt.tight_layout()
        plt.show()
        
        filtered_signals = signal.correlate(input_audio, np.reshape(sig, (-1, 1)), 'same', method='fft')
        roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
        envelopes = np.abs(signal.hilbert(roll_filt_sigs, axis=0))

        # fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
        # plt.suptitle('Envelopes')
        # for i in range(envelopes.shape[1]//2):
        #     for j in range(2):
        #         ax[i, j].plot(envelopes[:, 2*i+j])
        #         ax[i, j].set_title('Channel %d' % (2*i+j+1))
        #         ax[i, j].minorticks_on()
        #         ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        #         ax[i, j].grid()
        # plt.tight_layout()
        # plt.show()

        peaks = []
        for i in np.arange(nch):
            idxs, _ = signal.find_peaks(envelopes[:, i], prominence=10)
            peaks.append(idxs[0])
        
        peaks_array = np.array(peaks)
        
        earliest_peak = np.min(peaks_array)

        # analyzed_signals = input_audio[earliest_peak - 50:earliest_peak + 626]
        new_idxs = np.array([1, 0, 3, 2, 5, 4, 7, 6])
        analyzed_signals = roll_filt_sigs[earliest_peak - 50:earliest_peak + 50, new_idxs]
        peaks_array = peaks_array[new_idxs]
        print(peaks_array)
        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
        plt.suptitle('Analyzed Signals')
        for i in range(analyzed_signals.shape[1]//2):
            for j in range(2):
                ax[i, j].plot(analyzed_signals[:, 2*i+j])
                ax[i, j].axvline(peaks_array[2*i+j] - earliest_peak + 50, 0, 200, color='r', linestyle='--')
                ax[i, j].axvline(50, 0, 200, color='g', linestyle='-.')
                ax[i, j].set_title('Channel %d' % (2*i+j+1))
                ax[i, j].minorticks_on()
                ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                ax[i, j].grid()
        plt.tight_layout()
        plt.show()

        theta, p = das_filter(  
            analyzed_signals,
            fs=fs,
            nch=input_audio.shape[1],
            d=2.70e-3,
            bw=(low_freq, hi_freq),
            show=True,
            wlen=64
            )
        
        theta_bar = theta[np.argmax(p)]

        print(f'DoA: {theta_bar} [deg]')

        fig, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})

        ax2.plot(np.deg2rad(theta), 20*np.log10(p), color='r')
        ax2.set_title('Spatial Energy Distribution')
        ax2.set_xlabel('Angle [deg]')   
        # ax2.set_ylabel('Magnitude [dB]')
        # Shift axes by -90 degrees
        ax2.set_theta_offset(np.pi/2)
        # Limit theta between -90 and 90 degrees
        ax2.set_theta_direction(1)
        ax2.set_xlim(-np.pi/2, np.pi/2)
        ax2.set_xticks(np.deg2rad([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]))
        # ax2.set_yticks([-80, -60, -40])
        # ax.set_ylim(-20, 40)        
        ax2.grid(True)
        plt.show()
