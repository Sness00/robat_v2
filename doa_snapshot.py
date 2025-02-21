import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal
import queue
import time
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter
from capon import capon_method

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
        
def pow_two_pad_and_window(vec, fs, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.2)
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

if __name__ == "__main__":

    fs = 192000
    dur = 2e-3
    hi_freq = 55e3
    low_freq = 25e3

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
    sig = pow_two_pad_and_window(chirp, fs, show=False)

    silence_dur = 15 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = pow_two(np.concatenate((sig, silence_vec)))
    # full_sig = np.concatenate((sig, silence_vec))
    stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])

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
                        channels=(8, 2),
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
        filtered_signals = signal.correlate(valid_channels_audio, np.reshape(sig, (-1, 1)), 'full', method='fft')
        envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))

        mean_env = np.sum(envelopes, axis=1)/envelopes.shape[1]
        peaks, _ = signal.find_peaks(mean_env, prominence=10)

        furthest_peak = peaks[0]


        theta, p_capon = capon_method(filtered_signals[furthest_peak+192:], fs=fs, nch=filtered_signals.shape[1], d=0.003, bw=(low_freq, hi_freq))
        theta2, p_das2 = das_filter(filtered_signals[furthest_peak+192:], fs=fs, nch=filtered_signals.shape[1], d=0.003, bw=(low_freq, hi_freq))
        
        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
        for i in range(2):
            for j in range(4):
                ax[j, i].plot(filtered_signals[:, i*4+j])
                ax[j, i].vlines(np.array([furthest_peak, furthest_peak+192]), -20, 20, colors='r', linestyles='dashed')
                ax[j, i].set_title(f'Channel {i*4+j+1}')
        plt.tight_layout()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
        plt.suptitle('Capon Method vs DaS')
        ax1.plot(np.deg2rad(theta), p_capon)
        ax1.set_title('Capon Method Output')
        # Shift axes by -90 degrees
        ax1.set_theta_offset(np.pi/2)
        # Limit theta between -90 and 90 degrees
        ax1.set_theta_direction(1)
        ax1.set_xlim(-np.pi/2, np.pi/2)
        # ax.set_ylim(-20, 40)        
        ax1.grid(True)

        ax2.plot(np.deg2rad(theta2), p_das2)
        ax2.set_title('DaS Filter Output')
        # Shift axes by -90 degrees
        ax2.set_theta_offset(np.pi/2)
        # Limit theta between -90 and 90 degrees
        ax2.set_theta_direction(1)
        ax2.set_xlim(-np.pi/2, np.pi/2)
        # ax.set_ylim(-20, 40)        
        ax2.grid(True)
        plt.show()
