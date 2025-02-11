import sounddevice as sd
import time
from broadcast_pcmd3180 import activate_mics
import numpy as np
from matplotlib import pyplot as plt
import queue

def get_soundcard_instream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i

if __name__ == '__main__':
    
    fs = 44.1e3

    audio_in_data = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_in_data.put(indata.copy())

    
    activate_mics()
    start_time = time.time()

    stream = sd.InputStream(samplerate=fs, 
                            blocksize=1024, 
                            latency='high',
                            device=get_soundcard_instream(sd.query_devices()), 
                            channels=8, 
                            callback=callback)
    print('Stream started')
    with stream as s:
        while (time.time() - start_time) < 1:
            pass
    print('Stream closed')

    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
        input_audio = np.concatenate(all_input_audio)
    
    t = np.linspace(0, len(input_audio)/fs, len(input_audio))

    plt.figure()
    for i, audio in enumerate(np.transpose(input_audio)):
        plt.subplot(4, 2, i+1)
        plt.plot(t, audio)
        plt.title('Channel %d' % (i+1))
    plt.tight_layout()
    plt.show()