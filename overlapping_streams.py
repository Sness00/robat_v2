import os
import time
from datetime import datetime
import queue
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from thymiodirect import Thymio, Connection
from broadcast_pcmd3180 import activate_mics

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_soundcard_instream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')
        
if __name__ == '__main__':

    rec_dir = './recordings/'
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())


    fs = 192e3
    try:
        port = Connection.serial_default_port()
        now = datetime.now()
        th = Thymio(
            serial_port=port,
            on_connect=lambda node_id: print(f'\nThymio {node_id} is connected')
            )
        th.connect()
        robot_id = th.first_node()
        robot = th[robot_id]            
        # Delay to allow robot initialization of all variables
        time.sleep(1)

        robot['motor.left.target'] = 300        
        robot['motor.right.target'] = 300

        try:
            filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S-%f') + '.wav')

            with sf.SoundFile(filename, mode='x', samplerate=int(fs),
                            channels=8) as file:
                activate_mics()
                with sd.InputStream(samplerate=int(fs), device=get_soundcard_instream(sd.query_devices()),
                                    channels=8, callback=callback):
                    print('#' * 80)
                    print('press Ctrl+C to stop the recording')
                    print('#' * 80)
                    while True:
                        file.write(q.get())

        except KeyboardInterrupt:
            print('\nRecording finished: ' + repr(filename))
            robot['motor.left.target'] = 0
            robot['motor.right.target'] = 0
            th.disconnect()

            y, fs = sf.read(filename)
            dB_rms = 20*np.log10(np.mean(np.std(y, axis=0)))

            print('Thymio eigen-noise %.2f [dB]' % dB_rms)

            t = np.linspace(0, len(y)/fs, len(y))
            fig, ax = plt.subplots(y.shape[1]//2, 2, sharex=True, sharey=True)
            plt.suptitle('Channel Envelopes')
            for i in range(y.shape[1]//2):
                for j in range(2):
                    ax[i, j].plot(t, y[:, 2*i+j])
                    ax[i, j].set_title('Channel %d' % (2*i+j+1))
                    ax[i, j].minorticks_on()
                    ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                    ax[i, j].grid()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print('\nEncountered exception: ', e)