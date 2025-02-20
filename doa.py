import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal
import queue
import time
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter_v2
from matplotlib.animation import FuncAnimation

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
    try:
        def update(frame):
            stream = sd.Stream(samplerate=fs,
                        blocksize=0, 
                        device=soundcard, 
                        channels=(8, 2),
                        callback=callback,
                        latency='low')
            with stream:
                while stream.active:
                    pass
            global current_frame
            current_frame = 0
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

                # fig, axs = plt.subplots(8, 1, sharex=True, sharey=True)
                # peaks_array = np.array(peaks)
                # for i in range(8):
                #     axs[i].plot(filtered_signals[:, i])
                #     axs[i].vlines(np.array([furthest_peak, furthest_peak+70, 3500]), ymin=-10, ymax=10, colors='red')
                #     axs[i].set_title('Matched Filter Channel %d' % (i+1))
                #     axs[i].grid(True)
                # plt.tight_layout()
                # plt.show()

                theta2, p_das2 = das_filter_v2(filtered_signals[furthest_peak+70:furthest_peak+70+384, ], fs=fs, nch=filtered_signals.shape[1], d=0.003, bw=(low_freq, hi_freq))
                if max(p_das2) > 0.005:
                    theta_hat = np.argmax(p_das2)
                    print('Estimated DoA: %.2f [deg]' % theta2[theta_hat])
                else:
                    print('No DoA detected')
                line.set_ydata(20*np.log10(p_das2))
                return line,


        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_title('DaS Filter Output')
        # Shift axes by -90 degrees
        ax.set_theta_offset(np.pi/2)
        # Limit theta between -90 and 90 degrees
        ax.set_theta_direction(-1)
        ax.set_xlim(-np.pi/2, np.pi/2)
        ax.set_ylim(-20, 40)        
        ax.grid(True)
        line, = ax.plot(np.linspace(-np.pi/2, np.pi/2, 73), 0*np.sin(np.linspace(-np.pi/2, np.pi/2, 73)))
        ani = FuncAnimation(fig, update, frames=range(10), blit=False, interval=40)
        plt.show()

    except KeyboardInterrupt:
        print('Terminated by user')
    