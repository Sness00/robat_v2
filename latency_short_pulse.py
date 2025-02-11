#%%
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np 
import scipy.signal as signal
import queue
from smbus2 import SMBus
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def start_mics():
    with SMBus(1) as bus:
        if bus.read_byte_data(int('4E', 16), int('75', 16)) != int('60', 16):
            bus.write_byte_data(int('4E', 16), int('2', 16), int('81', 16))
            time.sleep(1e-3)
            bus.write_byte_data(int('4E', 16), int('7', 16), int('60', 16))
            bus.write_byte_data(int('4E', 16), int('B', 16), int('0', 16))
            bus.write_byte_data(int('4E', 16), int('C', 16), int('20', 16))
            bus.write_byte_data(int('4E', 16), int('22', 16), int('41', 16))
            bus.write_byte_data(int('4E', 16), int('2B', 16), int('40', 16))
            bus.write_byte_data(int('4E', 16), int('73', 16), int('C0', 16))
            bus.write_byte_data(int('4E', 16), int('74', 16), int('C0', 16))
            bus.write_byte_data(int('4E', 16), int('75', 16), int('60', 16))

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
        
def pow_two_pad_and_window(vec, show=False):
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
#%%
if __name__ == "__main__":

    # Load and resample at 192kHz the test audio
    x, fs = librosa.load('./1-80k_3ms.wav', sr=192000)
    sig = pow_two_pad_and_window(x, show=False)
    output_sig = np.pad(sig, (0, int(0.1*fs)))
    dur = len(output_sig) / fs

    output_sig = np.float32(np.reshape(output_sig, (-1, 1)))
    
    # Queue to store incoming audio data
    audio_in_data = queue.Queue()
    

# blocksize = 0, latency = 'low', output before input

    # Stream callback function
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
            raise sd.CallbackStop()
        current_frame += chunksize

    # Initialize and power on mics array
    start_mics()

    # Create stream
    stream = sd.Stream(samplerate=fs,
                       blocksize=0, 
                       device=get_soundcard_iostream(sd.query_devices()),
                       channels=(8, 1),
                       callback=callback,
                       latency=0.001
                       )
    
    # Little pause to let the soundcard settle
    time.sleep(0.5)
    # Run stream until playback stops
    with stream:
        print('Stream started')
        while stream.active:
            pass
        print('Stream ended')
    # Transfer input data from queue to an array
    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
    input_audio = np.concatenate(all_input_audio)

    # Save channel 2 recorded audio, as a reference to work on. Also channels 3, 6 and 7 could work
    rec_audio = input_audio[:, 3].reshape(-1, 1)
    if len(output_sig) > len(rec_audio):
        output_sig = output_sig[0:len(rec_audio)]
    elif len(output_sig) < len(rec_audio):
        output_sig = np.pad(output_sig, ((0, len(rec_audio) - len(output_sig)), (0, 0)))

#%%
    # Cross correlate recorded audio and test audio and find its absolute maximum as an estimate of audio latency
    cc_mod = signal.correlate(rec_audio, output_sig, 'full', method='fft')
    cc_mod_max_idx = np.argmax(np.abs(cc_mod)) - len(output_sig)
    lag_mod = (cc_mod_max_idx) / fs*1000
    print('Estimated latency:', lag_mod, '[ms]')

    # Print test audio and demodulated audio32
    # t = np.linspace(0, len(rec_audio)/fs, len(rec_audio))
    # plt.figure()
    # aa = plt.subplot(211) 
    # plt.plot(t, output_sig)
    # plt.title('Original Signal')
    # plt.subplot(212)
    # plt.plot(t, rec_audio)
    # plt.title('Recorded Signal')
    # plt.tight_layout()
    # plt.show()