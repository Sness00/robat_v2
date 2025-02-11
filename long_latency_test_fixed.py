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
        
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    sos = signal.tf2sos(b, a)
    y = signal.sosfilt(sos, data)
    return y

#%%
if __name__ == "__main__":

    # Load and resample at 192kHz the test audio
    x, fs = librosa.load('./test_audio.mp3', sr=192000)
    dur = len(x) / fs
    t = np.linspace(0, dur, len(x))

    # Modulate in ultrasonic band to exploit the transducer's frequency response
    fc = 45e3
    carrier = np.cos(2*np.pi*fc*t)
    mod_sig = x * carrier

    # Output signal
    output_sig = np.float32(np.reshape(mod_sig, (-1, 1)))

    # Queue to store incoming audio data
    audio_in_data = queue.Queue()
    
#%% 
# blocksize = 0, latency = 'low', output before input

    # Stream callback function
    
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
#%%
    # Create stream
    stream = sd.Stream(samplerate=fs,
                       blocksize=0, 
                       device=get_soundcard_iostream(sd.query_devices()),
                       channels=(8, 1),
                       callback=callback,
                       latency=0.004
                       )
    
    # Little pause to let the soundcard settle
    time.sleep(0.5)
    current_frame = 0
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
    rec_audio = input_audio[0:len(carrier), 2]

    # Demodulate recorded audio
    demod_audio = butter_lowpass_filter(2 * rec_audio * carrier, 48000, fs, 1)

    # Cross correlate recorded audio and test audio and find its absolute maximum as an estimate of audio latency
    cc_mod = signal.correlate(rec_audio, mod_sig, 'full')
    cc_mod_max_idx = np.argmax(np.abs(cc_mod)) - len(carrier)
    lag_mod = (cc_mod_max_idx) / fs*1000
    print('Estimated latency from modulated signals: %.2f [ms]' % (lag_mod))

    cc = signal.correlate(demod_audio, x, 'full')
    cc_max_idx = np.argmax(np.abs(cc)) - len(carrier)
    lag = (cc_max_idx) / fs*1000
    print('Estimated latency from base band signals: %.2f [ms]' % (lag))

#%%
    # Print test audio and demodulated audio
    # plt.figure()
    # aa = plt.subplot(311) 
    # plt.plot(t, x)
    # plt.title('Original Signal')
    # plt.subplot(313, sharex=aa, sharey=aa)
    # plt.plot(t, demod_audio)
    # plt.title('Demodulated Signal')
    # plt.subplot(312, sharex=aa, sharey=aa)
    # plt.plot(t, rec_audio)
    # plt.title('Recorded Signal')
    # plt.tight_layout()
    # plt.show()

#%%
    # sf.write('demod_audio.wav', demod_audio, fs)