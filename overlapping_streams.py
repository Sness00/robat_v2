import os
import threading
import traceback
from datetime import datetime
import queue
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import sounddevice as sd
import soundfile as sf
from soundfile import SoundFile, SEEK_END
from broadcast_pcmd3180 import activate_mics
from sonar import sonar
from das_v2 import das_filter
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_soundcard_instream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')


def get_soundcard_outstream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')

# def get_soundcard_iostream(device_list):
#     for i, each in enumerate(device_list):
#         dev_name = each['name']
#         asio_in_name = 'MCHStreamer' in dev_name
#         if asio_in_name:
#             return (i, i)
#     raise ValueError('No soundcard found')

def pow_two_pad_and_window(vec):
    window = signal.windows.tukey(len(vec), alpha=0.3)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    return padded_windowed_vec/max(padded_windowed_vec)

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

def recording_thread_function(q):
    while True:
        # if not mutex.locked():
            file.write(q.get())

if __name__ == '__main__':

    save_audio = True
    rec_dir = './recordings/'
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)

    q = queue.Queue()

    def in_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())
        # rec_q.put(indata.copy())
    
    current_frame = 0
    def out_callback(outdata, frames, time, status):
        global current_frame
        if status:
            print(status)
        chunksize = min(len(output_sig) - current_frame, frames)
        outdata[:chunksize] = output_sig[current_frame:current_frame + chunksize]
        if chunksize < frames:
            outdata[chunksize:] = 0            
            raise sd.CallbackStop()
        current_frame += chunksize

    fs = 176.4e3
    dur = 3e-3
    hi_freq = 60e3
    low_freq = 15e3
    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)
    sig = pow_two_pad_and_window(chirp)

    silence_dur = 20 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = pow_two(np.concatenate((sig, silence_vec)))
    output_sig = np.float32(np.reshape(full_sig, (-1, 1)))
    
    now = datetime.now()
    activate_mics()
    filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S') + '.wav')

    try:
        with SoundFile(filename, mode='x+', samplerate=int(fs),
                        channels=8) as file:
           
            with sd.InputStream(samplerate=int(fs), device=get_soundcard_instream(sd.query_devices()),
                                channels=8, callback=in_callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                if save_audio:
                    # mutex = threading.Lock()
                    input_thread = threading.Thread(target=recording_thread_function, args=(q,), daemon=True)
                    input_thread.start()                    
                    while True:
                        curr_end = file.frames                    
                        stream = sd.OutputStream(samplerate=fs,
                                            blocksize=0,
                                            device=get_soundcard_outstream(sd.query_devices()),
                                            channels=1,
                                            callback=out_callback,
                                            latency='low')
                        with stream:
                            while stream.active:
                                pass
                        current_frame = 0
                        # mutex.acquire()
                        offset = file.frames - curr_end
                        # file.seek(curr_end)

                        # rec_audio = file.read(offset)
                        rec_audio = sf.read(filename, start=curr_end, stop=curr_end+offset)[0]
                        # print(rec_audio.shape)
                        # mutex.release()                        
                        dB_rms = 20 * np.log10(np.sqrt(np.mean(rec_audio**2))+1e-10)
                    
                        #     # t_plot = np.linspace(0, rec_audio.shape[0]/fs, rec_audio.shape[0])
                        #     # fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
                        #     # plt.suptitle('Recorded Audio')
                        #     # for i in range(rec_audio.shape[1]//2):
                        #     #     for j in range(2):
                        #     #         ax[i, j].plot(t_plot, rec_audio[:, 2*i+j])
                        #     #         ax[i, j].set_title('Channel %d' % (2*i+j+1))
                        #     #         ax[i, j].minorticks_on()
                        #     #         ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                        #     #         ax[i, j].grid()
                        #     # plt.tight_layout()
                        #     # plt.show()
                        filtered_signals = signal.correlate(rec_audio, np.reshape(sig, (-1, 1)), 'same', method='fft')
                        # fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
                        # for i in range(filtered_signals.shape[1]//2):
                        #     for j in range(2):
                        #         ax[i, j].plot(filtered_signals[:, 2*i+j])
                        # plt.savefig(os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S') + '_rec.png'))
                        roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
                        distance, emission, echo = sonar(roll_filt_sigs, 100, fs)
                        theta, p = das_filter(
                                                roll_filt_sigs[echo - int(5e-4*fs):echo + int(5e-4*fs)], 
                                                fs=fs, nch=roll_filt_sigs.shape[1], d=2.70e-3, 
                                                bw=(low_freq, hi_freq)
                                            )
                        p_dB = 20*np.log10(p)
                    
                        if emission != echo:
                            doa_index = np.argmax(p_dB)
                            theta_hat = theta[doa_index]
                            print('\nDistance: %.1f [cm] | DoA: %.2f [deg]' % ((distance*100), theta_hat))
                        # rec_audio = file.read()
                    # rec_audio = []
                    # if not rec_q.empty():
                    #     while not rec_q.empty():
                    #         rec_audio.append(rec_q.get())
                    # rec_audio = np.concatenate(rec_audio)
                    # print(rec_audio.shape)

                    # save audio to file
                    # filename_1 = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S-%f') + '_rec.wav')
                    # if save_audio:                
                    #     sf.write(filename_1, rec_audio, int(fs))
                    # dB_rms = 20 * np.log10(np.sqrt(np.mean(rec_audio**2)))
                    # print('dB RMS: ', dB_rms)
                    
                    # filtered_signals = signal.correlate(rec_audio, np.reshape(sig, (-1, 1)), 'same', method='fft')
                    # roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
                    # distance = sonar(roll_filt_sigs, 0, fs)[0]
                    # print('\nDistance: %.1f' % (distance*100))
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(filename))