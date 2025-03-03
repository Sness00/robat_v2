import os
import argparse
import time
import queue
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sounddevice as sd
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter
from capon import capon_method
from music import music

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
        
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

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

def windower(a):
    window = signal.windows.tukey(len(a), alpha=0.2)
    if len(a.shape) > 1:
        window = np.reshape(window, (-1, 1))
    windowed_a = a * window
    return windowed_a

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Provide method and verbose option"
    )
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--verbose", required=False, type=bool, default=False)
    args = parser.parse_args()

    METHOD = args.method
    verbose = args.verbose

    fs = 192000
    nch = 8

    if METHOD == 'das':
        spatial_filter = das_filter
    elif METHOD == 'capon':
        spatial_filter = capon_method
    elif METHOD == 'music':
        spatial_filter = music
    
    processed_samples = 180
    discarded_samples = 480
    dur = 3e-3
    hi_freq = 45e3
    low_freq = 25e3

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
    sig = pow_two_pad_and_window(chirp, fs, show=False)

    silence_dur = 20 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = np.reshape(pow_two(np.concatenate((sig, silence_vec))), (-1, 1))
    stereo_sig = np.hstack([full_sig, full_sig])

    output_sig = np.float32(stereo_sig)

    audio_in_data = queue.Queue()

    current_frame = 0
    def callback(indata, outdata, frames, time, status):
        global current_frame
        if status:
            print(status)
        chunksize = min(len(output_sig) - current_frame, frames)
        outdata[:chunksize] = output_sig[current_frame:current_frame + chunksize]
        audio_in_data.put(indata.copy())
        if chunksize < frames:
            outdata[chunksize:] = 0
            raise sd.CallbackAbort()
        current_frame += chunksize

    activate_mics()
    soundcard = get_soundcard_iostream(sd.query_devices())
    
    # Little pause to let the soundcard settle
    time.sleep(0.5)

    stream = sd.Stream(samplerate=fs,
                        blocksize=0, 
                        device=soundcard, 
                        channels=(nch, 2),
                        callback=callback,
                        latency='low')
    with stream:
        while stream.active:
            pass

    # Transfer input data from queue to an array
    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
    input_audio = np.concatenate(all_input_audio)
    db_rms = 20*np.log10(np.std(input_audio))
    if db_rms < -50:
        print('Low output level. Replace amp battery')
    else:
        valid_channels_audio = input_audio
        filtered_signals = signal.correlate(valid_channels_audio, np.reshape(sig, (-1, 1)), 'same', method='fft')
        envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))

        mean_env = np.sum(envelopes, axis=1)/envelopes.shape[1]
        peaks, _ = signal.find_peaks(mean_env, prominence=1, distance=30)

        furthest_peak = peaks[0]

        if verbose:
            fig, ax = plt.subplots(nch//2, 2, sharex=True, sharey=True)
            plt.suptitle('Input audio')
            for i in range(2):
                for j in range(nch//2):
                    ax[j, i].plot(input_audio[:, i*nch//2+j])
                    ax[j, i].set_title(f'Channel {i*nch//2+j+1}')
                    ax[j, i].grid()
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(nch//2, 2, sharex=True, sharey=True)
            plt.suptitle('Filtered signals')
            for i in range(2):
                for j in range(nch//2):
                    ax[j, i].plot(filtered_signals[:, i*nch//2+j])
                    ax[j, i].vlines(np.array([furthest_peak, furthest_peak+discarded_samples, furthest_peak+discarded_samples+processed_samples]),
                                    -20, 20, colors='r', linestyles='dashed')
                    ax[j, i].set_title(f'Channel {i*nch//2+j+1}')
                    ax[j, i].grid()
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(nch//2, 2, sharex=True, sharey=True)
            plt.suptitle('Signals fed to spatial filter')
            for i in range(2):
                for j in range(nch//2):
                    ax[j, i].plot(windower(filtered_signals[furthest_peak+discarded_samples:furthest_peak+discarded_samples+processed_samples, i*nch//2+j]))
                    ax[j, i].set_title(f'Channel {i*nch//2+j+1}')
                    ax[j, i].grid()
            plt.tight_layout()
            plt.show()

        theta, p = spatial_filter(windower(filtered_signals[furthest_peak+discarded_samples:furthest_peak+discarded_samples+processed_samples]), fs=fs, nch=filtered_signals.shape[1], d=0.003, bw=(low_freq, hi_freq), wlen=64)
        theta_bar = theta[np.argmax(p)]

        print(f'DoA: {theta_bar} [deg]')

        fig, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})

        ax2.plot(np.deg2rad(theta), 20*np.log10(p))
        ax2.set_title('Pseudospectrum Magnitude\nSpatial Filter: ' + METHOD)
        ax2.set_xlabel('Angle [deg]')
        ax2.set_ylabel('Magnitude [dB]')
        # Shift axes by -90 degrees
        ax2.set_theta_offset(np.pi/2)
        # Limit theta between -90 and 90 degrees
        ax2.set_theta_direction(1)
        ax2.set_xlim(-np.pi/2, np.pi/2)
        # ax.set_ylim(-20, 40)        
        ax2.grid(True)
        plt.show()
