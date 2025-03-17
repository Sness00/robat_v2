import sounddevice as sd
import scipy.signal as signal
import numpy as np
from matplotlib import pyplot as plt
import time

def get_soundcard_outstream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')


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

if __name__ == '__main__':
    fs = 192000
    
    dur = 0.5 # s
    t = np.linspace(0, dur, int(dur*fs))
    chirp = signal.chirp(t, 15e3, t[-1], 2.5e3)

    sig = pow_two_pad_and_window(chirp, fs, show=False)

    output_sig = np.float32(sig.reshape(-1, 1))
    
    current_frame = 0
    def callback(outdata, frames, time, status):
        global current_frame
        if status:
            print(status)
        chunksize = min(len(output_sig) - current_frame, frames)
        outdata[:chunksize] = output_sig[current_frame:current_frame + chunksize]
        if chunksize < frames:
            outdata[chunksize:] = 0
            raise sd.CallbackAbort()
        current_frame += chunksize
    
    soundcard = get_soundcard_outstream(sd.query_devices())

    stream = sd.OutputStream(samplerate=fs,
                       blocksize=0, 
                       device=soundcard, 
                       channels=1,
                       callback=callback,
                       latency='low')
    
    time.sleep(0.5)

    with stream:
        print('Stream started')
        while stream.active:
            pass
    print('Stream closed')