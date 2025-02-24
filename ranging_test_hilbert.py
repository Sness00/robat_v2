import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal
import queue
import time
from broadcast_pcmd3180 import activate_mics

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
        
def pow_two_pad_and_window(vec, fs, show=False):
    window = signal.windows.hann(len(vec))
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

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, 60e3, t_tone[-1], 40e3)    
    sig = pow_two_pad_and_window(chirp, fs, show=False)

    silence_dur = 15 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = pow_two(np.concatenate((sig, silence_vec)))
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
        if chunksize < frames:
            outdata[chunksize:] = 0
            raise sd.CallbackAbort()
        current_frame += chunksize
        audio_in_data.put(indata.copy())

    activate_mics()
    soundcard = get_soundcard_iostream(sd.query_devices())

    stream = sd.Stream(samplerate=fs,
                       blocksize=0, 
                       device=soundcard, 
                       channels=(8, 2),
                       callback=callback,
                       latency='low')
    
    # Little pause to let the soundcard settle
    time.sleep(0.5)

    with stream:
        while stream.active:
            pass
    
    # Transfer input data from queue to an array
    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
    input_audio = np.concatenate(all_input_audio)

    # echoes = input_audio[:2650, :]

    # rms = 20*np.log10(np.std(echoes, axis=0))

    # print(rms)


    valid_channels_audio = input_audio
    filtered_signals = signal.correlate(valid_channels_audio, np.reshape(sig, (-1, 1)), 'full', method='fft')
    envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))

    
    peaks = []
    enough = True
    for i in np.arange(envelopes.shape[1]):
        idxs, _ = signal.find_peaks(envelopes[:, i], prominence=5, distance=30)
        if len(idxs) < 2:
            enough = False
        peaks.append(idxs[0:2])

    if not enough:
        print('Not enough peaks found')

    else:
        estimated_distances = []
        mean_dist = 0
        for i, p in enumerate(peaks):
            dist = (p[1] - p[0])/fs*343.0/2 + 0.025
            estimated_distances.append(dist)
            print('Estimated distance for channel', i+1, ':', '%.5f' % dist, '[m]')    
            mean_dist += dist
        mean_dist /= len(peaks)
        peaks_array = np.array(peaks)

        print('Estimated mean distance: %.5f' % mean_dist, '[m]')

    t_plot = np.linspace(0, envelopes.shape[0]/fs, envelopes.shape[0])
    plt.figure()
    aa = plt.subplot(421)
    if enough:
        plt.vlines(peaks_array[0, :]/fs, 0, max(envelopes[:, 0]), linestyles='dashed', colors='r')
    plt.plot(t_plot, envelopes[:, 0])
    plt.title('Envelope of Channel 1')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.grid()

    for i in np.arange(1, envelopes.shape[1]):
        plt.subplot(4, 2, i+1, sharex=aa, sharey=aa)
        if enough:
            plt.vlines(peaks_array[i, :]/fs, 0, max(envelopes[:, i]), linestyles='dashed', colors='r')
        plt.plot(t_plot, envelopes[:, i])
        plt.title('Envelope of Channel %d' %(i+1))
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.grid()

    plt.tight_layout()
    plt.show()
    
    mean_envelope = np.sum(envelopes, axis=1)/envelopes.shape[1]
    if enough:
        peaks, _ = signal.find_peaks(mean_envelope, prominence=5, distance=30)
        est_dist = (peaks[1] - peaks[0])/fs*343.0/2 + 0.025

    if enough:
        print('Estimated distance from the mean envelope:', '%.5f' % est_dist, '[m]')
    else:
        print('Not enough')

    plt.figure()
    if enough:
        plt.vlines(peaks[0:2]/fs, 0, max(mean_envelope), linestyles='dashed', colors='r')
    plt.plot(t_plot, mean_envelope)
    plt.title('Mean Envelope')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # t_plot = np.linspace(0, input_audio.shape[0]/fs, input_audio.shape[0])
    # plt.figure()
    # aa = plt.subplot(4, 2, 1)
    # plt.plot(t_plot, input_audio[:, 0])
    # plt.title('Channel 1 Audio' )
    # plt.grid()
    # for i in np.arange(1, input_audio.shape[1]):
    #     plt.subplot(4, 2, i+1, sharex=aa, sharey=aa)
    #     plt.plot(t_plot, input_audio[:, i])
    #     plt.title('Channel %d Audio' % (i+1))
    #     plt.grid()
    # plt.tight_layout()
    # plt.show()

    # t_plot = np.linspace(0, filtered_signals.shape[0]/fs, filtered_signals.shape[0])
    # plt.figure()
    # for i in np.arange(0, filtered_signals.shape[1]):
    #     plt.subplot(4, 2, i+1)
    #     plt.plot(t_plot, filtered_signals[:, i])
    #     plt.title('Matched Filter Channel %d' % (i+1))
    # plt.tight_layout()
    # plt.show()
        