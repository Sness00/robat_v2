import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal
import queue
import time
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter_v2

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
    hi_freq = 52.5e3
    low_freq = 27.5e3

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
    sig = pow_two_pad_and_window(chirp, fs, show=False)

    silence_dur = 10 # [ms]
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
    db_rms = 20*np.log10(np.std(input_audio))
    if db_rms < -600:
        print('Low output level. Replace amp battery')
    else:
        valid_channels_audio = input_audio
        filtered_signals = signal.correlate(valid_channels_audio, np.reshape(sig, (-1, 1)), 'full', method='fft')
        envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))

        # mean_env = np.sum(envelopes, axis=1)/envelopes.shape[1]
        # peaks, _ = signal.find_peaks(mean_env, prominence=10)

        # furthest_peak = peaks[0]

        # fig, axs = plt.subplots(8, 1, sharex=True, sharey=True)
        # peaks_array = np.array(peaks)
        # for i in range(8):
        #     axs[i].plot(filtered_signals[:, i])
        #     axs[i].vlines(np.array([furthest_peak, furthest_peak+70, 3500]), ymin=-10, ymax=10, colors='red')
        #     axs[i].set_title('Matched Filter Channel %d' % (i+1))
        #     axs[i].grid(True)
        # plt.tight_layout()
        # plt.show()

        # theta2, p_das2 = das_filter_v2(filtered_signals[furthest_peak+70:3500, ], fs=fs, nch=filtered_signals.shape[1], d=0.003, bw=(low_freq, hi_freq))

        # if max(p_das2) > 0.01:
        #     theta_hat = np.argmax(p_das2)
        
        # print('Estimated DoA: %d [deg]' % theta2[theta_hat])
        # plt.figure()
        # plt.plot(theta2, p_das2)
        # if max(p_das2) > 0.001:
        #     plt.vlines(theta2[theta_hat], ymin=0, ymax=max(p_das2)*1.1, colors='red')
        # plt.grid()
        # plt.title('Fast Implementation')
        # plt.tight_layout()
        # plt.show()

        # fig3, axs3 = plt.subplots(4, 2, sharex=True, sharey=True)
        # for i in range(8):
        #     axs3[i].plot(input_audio[:, i])
        #     axs3[i].set_title('Recorded Audio Channel %d' % (i+1))
        #     axs3[i].grid(True)
        # plt.tight_layout()
        # plt.show()
        # RecSignals = fft.fft(input_audio, n=2**int(np.ceil(np.log2(len(input_audio)))), axis=0)

        # FiltSignals = fft.fft(filtered_signals, n=2**int(np.ceil(np.log2(len(filtered_signals)))), axis=0)

        # freqRS = np.arange(0, len(RecSignals)//2) * fs/2/(len(RecSignals)//2) 
        # freqFS = np.arange(0, len(FiltSignals)//2) * fs/2/(len(FiltSignals)//2)

        # peaks, _ = signal.find_peaks(envelopes, prominence=8)
        t_rs = np.linspace(0, input_audio.shape[0]/fs, input_audio.shape[0])
        t_fs = np.linspace(0, filtered_signals.shape[0]/fs, filtered_signals.shape[0])
        plt.figure()
        plt.subplot(411)
        plt.specgram(input_audio[:, 0], NFFT=128, Fs=fs, noverlap=64)
        plt.title('Recorded Signal Spectrogram')
        plt.subplot(412)
        plt.plot(input_audio[:, 0])
        plt.grid()
        plt.title('Recorded Signal Time History')
        plt.subplot(413)
        plt.specgram(filtered_signals[:, 0], NFFT=128, Fs=fs, noverlap=64)
        plt.title('Matched Filter Signal Spectrogram')
        plt.subplot(414)
        plt.plot(filtered_signals[:, 0])
        plt.grid()
        plt.title('Matched Filtered Signal Time History')
        plt.tight_layout()
        plt.show()

        # plt.figure()
        # plt.subplot(121)
        # plt.plot(freqRS, np.abs(RecSignals[0:len(RecSignals)//2, 0]))
        # plt.grid()
        # plt.title('Recorded Signal Spectrum Amplitude')
        # plt.subplot(122)
        # plt.plot(freqFS, np.abs(FiltSignals[0:len(FiltSignals)//2, 0]))
        # plt.grid()
        # plt.title('Matched Filtered Signal Spectrum Amplitude')
        # plt.tight_layout()
        # plt.show()
    