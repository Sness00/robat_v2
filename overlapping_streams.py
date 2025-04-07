import os
import time
from datetime import datetime
import queue
import numpy as np
from scipy import signal
import sounddevice as sd
import soundfile as sf
from broadcast_pcmd3180 import activate_mics

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_soundcard_instream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
    raise ValueError('No soundcard found')

def pow_two_pad_and_window(vec):
    window = signal.windows.tukey(len(vec), alpha=0.3)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    return padded_windowed_vec/max(padded_windowed_vec)

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

if __name__ == '__main__':

    rec_dir = './recordings/'
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)

    q = queue.Queue()

    def in_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())
    
    fs = 192e3
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

    audio_in_data = queue.Queue()

    current_frame = 0

    def io_callback(indata, outdata, frames, time, status):
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


    
    now = datetime.now()

    try:
        filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S') + '.wav')

        with sf.SoundFile(filename, mode='x', samplerate=int(fs),
                        channels=8) as file:
            activate_mics()
            with sd.InputStream(samplerate=int(fs), device=get_soundcard_instream(sd.query_devices()),
                                channels=8, callback=in_callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    file.write(q.get())
                    stream = sd.Stream(samplerate=fs,
                        blocksize=0,
                        device=get_soundcard_iostream(sd.query_devices()),
                        channels=(8, 1),
                        callback=io_callback,
                        latency='low')
                    with stream:
                        while stream.active:
                            pass
                    current_frame = 0
                    all_input_audio = []
                    while not audio_in_data.empty():
                        all_input_audio.append(audio_in_data.get())

    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(filename))
