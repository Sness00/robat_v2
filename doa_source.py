import os
import time
from datetime import datetime
import queue
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter

def get_soundcard_instream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')

def windower(a):
    window = signal.windows.tukey(len(a), alpha=0.2)
    if len(a.shape) > 1:
        window = np.reshape(window, (-1, 1))
    windowed_a = a * window
    return windowed_a

if __name__ == "__main__":

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    fs = 192000
    C_AIR = 343
    nch = 8

    first_sample = int(fs*0.1)
    last_sample = first_sample + int(3e-3*fs)
    sig, _ = sf.read('playback_sweeps_fast_rising.wav', start=first_sample, stop=last_sample)
    
    field_range = 50e-2
    discarded_samples = int(np.floor((field_range*2)/C_AIR*fs)) - 60
    processed_samples = 512

    hi_freq = 95e3
    low_freq = 15e3

    rec_dir = './recordings/'
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    audio_in_data = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_in_data.put(indata.copy())

    activate_mics()

    try:        
        now = datetime.now()
        filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S-%f') + '.wav')
        with sf.SoundFile(filename, mode='x', samplerate=int(fs),
                            channels=8) as file:
            stream = sd.InputStream(samplerate=fs,
                                device=get_soundcard_instream(sd.query_devices()), 
                                channels=8, 
                                callback=callback,
                                latency='low',
                                blocksize=0 
                                )
            print('Stream started')
            with stream:
                while True:
                    file.write(audio_in_data.get())

    except KeyboardInterrupt:
        print('Stream closed')

        input_audio, _ = sf.read(filename)
        theta, p = das_filter(  
            windower(input_audio),
            fs=fs,
            nch=input_audio.shape[1],
            d=2.70e-3,
            bw=(low_freq, hi_freq),
            show=True,
            wlen=128
            )
        
        theta_bar = theta[np.argmax(p)]

        print(f'DoA: {theta_bar} [deg]')

        fig, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})

        ax2.plot(np.deg2rad(theta), 20*np.log10(p), color='r')
        ax2.set_title('Spatial Energy Distribution')
        ax2.set_xlabel('Angle [deg]')   
        # ax2.set_ylabel('Magnitude [dB]')
        # Shift axes by -90 degrees
        ax2.set_theta_offset(np.pi/2)
        # Limit theta between -90 and 90 degrees
        ax2.set_theta_direction(1)
        ax2.set_xlim(-np.pi/2, np.pi/2)
        ax2.set_xticks(np.deg2rad([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]))
        # ax2.set_yticks([-80, -60, -40])
        # ax.set_ylim(-20, 40)        
        ax2.grid(True)
        plt.show()
