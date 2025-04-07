import os
import threading
import time
from datetime import datetime
import queue
import numpy as np
from scipy import signal
import sounddevice as sd
import soundfile as sf
from broadcast_pcmd3180 import activate_mics
from sonar import sonar

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
    return padded_windowed_vec/max(padded_windowed_vec)/10

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

def recording_thread_function(filename, fs, q):
    with sf.SoundFile(filename, mode='x', samplerate=int(fs),
                        channels=8) as file:
           
        with sd.InputStream(samplerate=int(fs), device=get_soundcard_instream(sd.query_devices()),
                            channels=8, callback=in_callback):
            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                file.write(q.get())

if __name__ == '__main__':

    save_audio = True
    rec_dir = './recordings/'
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)

    q = queue.Queue()
    rec_q = queue.Queue()

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
            current_frame = 0
            raise sd.CallbackAbort()
        current_frame += chunksize

    # def io_callback(indata, outdata, frames, time, status):
    #     global current_frame
    #     if status:
    #         print(status)
    #     chunksize = min(len(output_sig) - current_frame, frames)
    #     outdata[:chunksize] = output_sig[current_frame:current_frame + chunksize]
    #     if chunksize < frames:
    #         outdata[chunksize:] = 0
    #         raise sd.CallbackAbort()
    #     current_frame += chunksize
    #     rec_q.put(indata.copy())

    fs = 176.4e3
    dur = 3e-3
    hi_freq = 20e3
    low_freq = 1e3
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
        if save_audio:
            input_thread = threading.Thread(target=recording_thread_function, args=(filename, fs, q), daemon=True)
            input_thread.start()
        
    
            while True:
                stream = sd.OutputStream(samplerate=fs,
                                    blocksize=0,
                                    device=get_soundcard_outstream(sd.query_devices()),
                                    channels=1,
                                    callback=out_callback,
                                    latency='low')
                with stream:
                    while stream.active:
                        pass
                
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
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(filename))