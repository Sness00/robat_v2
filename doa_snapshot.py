import os
import time
from datetime import datetime
import queue
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import sounddevice as sd
import soundfile as sf
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter
from capon import capon_method
from music_v2 import music

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
    raise ValueError('No soundcard found')
        
def pow_two_pad_and_window(vec, fs, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.3)
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

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    fs = 192000
    C_AIR = 343
    nch = 8

    METHOD = 'music'    
    if METHOD == 'das':
        spatial_filter = das_filter
    elif METHOD == 'capon':
        spatial_filter = capon_method
    elif METHOD == 'music':
        spatial_filter = music
    
    verbose = False
    field_range = 50e-2
    discarded_samples = int(np.floor((field_range*2)/C_AIR*fs)) - 60
    print(discarded_samples)
    processed_samples = 512
    dur = 3e-3
    hi_freq = 60e3
    low_freq = 20e3

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

    if db_rms > -50:
        valid_channels_audio = input_audio
        rec_dir = './doa_data/'
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        now = datetime.now()
        filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S') + '.wav')
        sf.write(filename, valid_channels_audio, fs)
        print('\nRecording saved in %s' % filename)
        filtered_signals = signal.correlate(valid_channels_audio, np.reshape(sig, (-1, 1)), 'same', method='fft')
        roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
        envelopes = np.abs(signal.hilbert(roll_filt_sigs, axis=0))

        mean_env = np.sum(envelopes, axis=1)/envelopes.shape[1]
        peaks, _ = signal.find_peaks(mean_env, prominence=10)

        furthest_peak = peaks[0]

        if verbose:

            fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
            plt.suptitle('Channel Envelopes')
            for i in range(envelopes.shape[1]//2):
                for j in range(2):
                    ax[i, j].plot(envelopes[:, 2*i+j])
                    ax[i, j].vlines(np.array([furthest_peak, furthest_peak+discarded_samples, furthest_peak+discarded_samples+processed_samples]), 0, 20, colors='r', linestyles='dashed')
                    ax[i, j].set_title('Channel %d' % (2*i+j+1))
                    ax[i, j].minorticks_on()
                    ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                    ax[i, j].grid()
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(nch//2, 2, sharex=True, sharey=True)
            plt.suptitle('Input Audio')
            for i in range(2):
                for j in range(nch//2):
                    ax[j, i].plot(input_audio[:, i*nch//2+j])
                    ax[j, i].vlines(np.array([furthest_peak, furthest_peak+discarded_samples, furthest_peak+discarded_samples+processed_samples]),
                                    -0.75, 0.75, colors='r', linestyles='dashed')
                    ax[j, i].set_title(f'Channel {i*nch//2+j+1}')
                    ax[j, i].grid()
            plt.tight_layout()
            plt.show()

        theta, p = spatial_filter(
            roll_filt_sigs[furthest_peak+discarded_samples:furthest_peak+discarded_samples+processed_samples],
                                    fs=fs, 
                                    nch=roll_filt_sigs.shape[1], 
                                    d=2.70e-3, 
                                    bw=(low_freq, hi_freq), 
                                    show=True, 
                                    wlen=128
                                    )
        p_dB = 10*np.log10(p)
        theta_bar = theta[np.argmax(p_dB)]
        doas = theta[signal.find_peaks(p_dB, prominence=10)[0]]
        print(f'\nDoA: {theta_bar} [deg]')

        fig, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})

        ax2.plot(np.deg2rad(theta), p_dB, color='r')
        ax2.vlines(np.deg2rad(doas), np.min(p_dB), np.max(p_dB), colors='g', linestyles='dashed')
        ax2.set_title('Spatial Energy Distribution')
        ax2.set_theta_offset(np.pi/2)
        ax2.set_xlim(-np.pi/2, np.pi/2)
        ax2.set_xticks(np.deg2rad([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]))    
        ax2.grid(True)
        plt.show()
    else:
        print('\nLow input level. Dead battery?')
