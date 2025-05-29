# %%
import os
import sounddevice as sd
import soundfile as sf
from broadcast_pcmd3180 import activate_mics
import numpy as np
from matplotlib import pyplot as plt
import queue
from scipy import signal
import time
from datetime import datetime

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

if __name__ == '__main__':
    
    fs = 192e3

    dur = 3e-3
    hi_freq = 90e3
    low_freq = 10e3

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
    sig = pow_two_pad_and_window(chirp, fs, show=False)

    silence_dur = 30 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = np.reshape(pow_two(np.concatenate((sig, silence_vec))), (-1, 1))
    stereo_sig = np.hstack([full_sig, full_sig])

    output_sig = np.float32(stereo_sig)

    audio_in_data = queue.Queue()

    

    activate_mics()
    soundcard = get_soundcard_iostream(sd.query_devices())
    
    # Little pause to let the soundcard settle
    time.sleep(0.5)
# %%
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

    stream = sd.Stream(samplerate=fs,
                        blocksize=0, 
                        device=soundcard, 
                        channels=(1, 2),
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
    filtered_signals = signal.correlate(input_audio, np.reshape(sig, (-1, 1)), 'same', method='fft')
    roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
    envelopes = np.abs(signal.hilbert(roll_filt_sigs, axis=0))
    mean_envelope = np.sum(envelopes, axis=1)/envelopes.shape[1]
        # %%
    plt.figure()
    plt.specgram(input_audio[int(0.01*fs):int(0.027*fs), 0], NFFT=128, noverlap=64, Fs=fs, sides='onesided', scale_by_freq=False, mode='psd', scale='dB', cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    # plt.colorbar()
    plt.show()
    # %%
    fig = plt.figure()
    plt.plot(roll_filt_sigs[3200:3500], color='black', linewidth=0.75)
    plt.plot(envelopes[3200:3500], color='red')
    plt.legend(['Matched Filter Output', 'Envelope'])
    plt.show()
    # fig.savefig('envelope.png', transparent=True)
# %%
    rec_dir = './recordings/'
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    
    now = datetime.now()
    filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S') + '.wav')

    sf.write(filename, input_audio, samplerate=int(fs))