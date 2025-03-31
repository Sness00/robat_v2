
import os
import sounddevice as sd
import soundfile as sf
from broadcast_pcmd3180 import activate_mics
import numpy as np
from matplotlib import pyplot as plt
import queue
from datetime import datetime

def get_soundcard_instream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')

if __name__ == '__main__':

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    fs = 192e3

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
        filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S') + '.wav')
        with sf.SoundFile(filename, mode='x', samplerate=int(fs),
                            channels=8) as file:
            stream = sd.InputStream(samplerate=fs,
                                device=get_soundcard_instream(sd.query_devices()), 
                                channels=8, 
                                callback=callback,
                                latency='low',
                                blocksize=0 
                                )
            print('\nStream started')
            with stream:
                while True:
                    file.write(audio_in_data.get())

    except KeyboardInterrupt:
        print('\nStream closed')

        input_audio, _ = sf.read(filename)
        
        t = np.linspace(0, len(input_audio)/fs, len(input_audio))

        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
        plt.suptitle('Recorded Audio')
        for i in range(input_audio.shape[1]//2):
            for j in range(2):
                ax[i, j].plot(t, input_audio[:, 2*i+j])
                ax[i, j].set_title('Channel %d' % (2*i+j+1))
                ax[i, j].minorticks_on()
                ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                ax[i, j].grid()
        plt.tight_layout()
        plt.show()