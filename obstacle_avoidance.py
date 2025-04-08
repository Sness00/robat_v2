import os
import sys
import threading
import traceback
import time
from datetime import datetime
import queue
import random
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import sounddevice as sd
import soundfile as sf
from thymiodirect import Thymio, Connection
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter
from capon import capon_method
from sonar import sonar

def get_soundcard(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return i
    raise ValueError('No soundcard found')

def pow_two_pad_and_window(vec, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.3)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(padded_windowed_vec))
        plt.figure()
        plt.plot(t, padded_windowed_vec)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)*0.8

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

def angle_to_time(angle, speed):
    A = 612.33
    B = -0.94
    t = A*speed**B    
    return t * abs(angle) / 360

def recording_thread_function(q):
    while True:
        file.write(q.get())

if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    speed = 200
    rot_speed = 200
    lateral_threshold = 30000
    ground_threshold = 10000
    air_threshold = 10
    output_threshold = -50 # [dB]
    distance_threshold = 30 # [cm]

    save_audio = True
    rec_dir = './recordings/'
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)

    q = queue.Queue()

    def in_callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())
    
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

    C_AIR = 343
    min_distance = 10e-2
    discarded_samples = int(np.floor((min_distance*2)/C_AIR*fs))

    METHOD = 'das'
    if METHOD == 'das':
        spatial_filter = das_filter
    elif METHOD == 'capon':
        spatial_filter = capon_method

    dur = 3e-3
    hi_freq = 60e3
    low_freq = 15e3
    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)
    sig = pow_two_pad_and_window(chirp, show=False)

    silence_dur = 20 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = pow_two(np.concatenate((sig, silence_vec)))
    output_sig = np.float32(np.reshape(full_sig, (-1, 1)))

    device = get_soundcard(sd.query_devices())

    now = datetime.now()
    activate_mics()
    filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S') + '.wav')

    try:
        port = Connection.serial_default_port()
        try:
            th = Thymio(serial_port=port,
            on_connect=lambda node_id: print(f'\nThymio {node_id} is connected'))
            th.connect()
            robot = th[th.first_node()]            
            # Delay to allow robot initialization of all variables
            time.sleep(1)
            
            robot['motor.left.target'] = speed
            robot['motor.right.target'] = speed
            try:
                with sf.SoundFile(filename, mode='x+', samplerate=int(fs),
                                channels=8) as file:
                
                    with sd.InputStream(samplerate=int(fs), device=device,
                                        channels=8, callback=in_callback):
                        if save_audio:
                            input_thread = threading.Thread(target=recording_thread_function, args=(q,), daemon=True)
                            input_thread.start()                    
                            while True:
                                # Robot left the ground
                                if (robot['prox.ground.reflected'][0] < air_threshold or robot['prox.ground.reflected'][1] < air_threshold):
                                    print('\nRobot left the ground')
                                    raise KeyboardInterrupt
                                # Left ground sensor
                                elif robot['prox.ground.reflected'][0] > ground_threshold:
                                    robot['leds.bottom.left'] = [255, 0, 0]
                                    robot['leds.bottom.right'] = [255, 0, 0]
                                    robot['motor.left.target'] = rot_speed
                                    robot['motor.right.target'] = -rot_speed
                                    while robot['prox.ground.reflected'][0] > ground_threshold:
                                        pass
                                    robot['leds.bottom.left'] = [0, 0, 0]
                                    robot['leds.bottom.right'] = [0, 0, 0]
                                    robot['motor.left.target'] = speed
                                    robot['motor.right.target'] = speed
                                # Right ground sensor
                                elif robot['prox.ground.reflected'][1] > ground_threshold:
                                    robot['leds.bottom.left'] = [255, 0, 0]
                                    robot['leds.bottom.right'] = [255, 0, 0]
                                    robot['motor.left.target'] = -rot_speed
                                    robot['motor.right.target'] = rot_speed
                                    while robot['prox.ground.reflected'][1] > ground_threshold:
                                        pass
                                    robot['motor.left.target'] = speed
                                    robot['motor.right.target'] = speed
                                    robot['leds.bottom.left'] = [0, 0, 0]
                                    robot['leds.bottom.right'] = [0, 0, 0]

                                curr_end = file.frames                    
                                stream = sd.OutputStream(samplerate=fs,
                                                    blocksize=0,
                                                    device=device,
                                                    channels=1,
                                                    callback=out_callback,
                                                    latency='low')
                                with stream:
                                    while stream.active:
                                        pass
                                current_frame = 0
                                offset = file.frames - curr_end
                                if offset > 0:            
                                    input_audio = sf.read(filename, start=curr_end, stop=curr_end+offset)[0] 

                                    dB_rms = 20*np.log10(np.mean(np.std(input_audio, axis=0))) # Battery is dead or not connected
                                    
                                    if dB_rms > output_threshold:
                                        filtered_signals = signal.correlate(input_audio, np.reshape(sig, (-1, 1)), 'same', method='fft')
                                        roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
                                        try:
                                            distance, direct_path, obst_echo = sonar(roll_filt_sigs, discarded_samples, fs)
                                            distance = distance*100 # [m] to [cm]                                    

                                            theta, p = spatial_filter(
                                                                        roll_filt_sigs[obst_echo - int(5e-4*fs):obst_echo + int(5e-4*fs)], 
                                                                        fs=fs, nch=roll_filt_sigs.shape[1], d=2.70e-3, 
                                                                        bw=(low_freq, hi_freq)
                                                                    )
                                            p_dB = 20*np.log10(p)
                                            
                                            if direct_path != obst_echo:
                                                doa_index = np.argmax(p_dB)
                                                theta_hat = theta[doa_index]
                                                print('\nDistance: %.1f [cm] | DoA: %.2f [deg]' % (distance, theta_hat))

                                            if distance < distance_threshold and distance > 0:
                                                robot['leds.bottom.left'] = [0, 255, 0]
                                                robot['leds.bottom.right'] = [0, 255, 0]

                                                if theta_hat > 0:
                                                    robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 255, 255]
                                                    direction = 'r'
                                                    t_rot = angle_to_time(theta_hat, rot_speed)
                                                elif theta_hat < 0:
                                                    robot['leds.circle'] = [0, 255, 255, 0, 0, 0, 0, 0]
                                                    direction = 'l'
                                                    t_rot = angle_to_time(theta_hat, rot_speed)
                                                else:
                                                    robot['leds.circle'] = [255, 0, 0, 0, 0, 0, 0, 0]
                                                    direction = random.choice(['l', 'r'])
                                                    t_rot = angle_to_time(180, rot_speed)

                                                if direction == 'l':
                                                    robot['motor.left.target'] = -rot_speed
                                                    robot['motor.right.target'] = rot_speed
                                                else:
                                                    robot['motor.left.target'] = rot_speed
                                                    robot['motor.right.target'] = -rot_speed
                                                
                                                time.sleep(t_rot)

                                                robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 0, 0]
                                                
                                                robot['leds.bottom.left'] = [0, 0, 0]
                                                robot['leds.bottom.right'] = [0, 0, 0]
                                                robot['motor.left.target'] = speed
                                                robot['motor.right.target'] = speed
                                        except ValueError:
                                            pass
                                else:
                                    print('\nLow output level. Dead battery?')
                    
                                #Left proximity sensor
                                if robot['prox.horizontal'][0] > lateral_threshold:
                                    robot['leds.bottom.left'] = [0, 0, 255]
                                    robot['motor.left.target'] = rot_speed
                                    robot['motor.right.target'] = -rot_speed
                                    while robot['prox.horizontal'][0] > lateral_threshold:
                                        pass
                                    robot['leds.bottom.left'] = [0, 0, 0]
                                    robot['motor.left.target'] = speed
                                    robot['motor.right.target'] = speed
                                # Right proximity sensor
                                elif robot['prox.horizontal'][4] > lateral_threshold-1000:
                                    robot['leds.bottom.right'] = [0, 0, 255]
                                    robot['motor.left.target'] = -rot_speed
                                    robot['motor.right.target'] = rot_speed
                                    while robot['prox.horizontal'][4] > lateral_threshold:
                                        pass
                                    robot['leds.bottom.right'] = [0, 0, 0]
                                    robot['motor.left.target'] = speed
                                    robot['motor.right.target'] = speed
            except Exception as e:
                print('\nException encountered:', e)
                traceback.print_exc()
                robot['motor.left.target'] = 0
                robot['motor.right.target'] = 0
                robot['leds.bottom.left'] = 0
                robot['leds.bottom.right'] = 0
                robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 0, 0]
                time.sleep(1)
                try:
                    th.disconnect()
                except Exception as e:
                    print('\nException encountered:', e)
                    traceback.print_exc()                    
            except KeyboardInterrupt:            
                print('\nTerminated by user')
                robot['motor.left.target'] = 0
                robot['motor.right.target'] = 0
                robot['leds.bottom.left'] = 0
                robot['leds.bottom.right'] = 0
                robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 0, 0]
                time.sleep(1)
                try:                
                    th.disconnect()
                except Exception as e:
                    print('\nException encountered:', e)
                    traceback.print_exc()
                finally:
                    print('\nExiting')
                    sys.exit(0)

        except Exception as e:
            print('\nException encountered:', e)
            traceback.print_exc()
    except Exception as e:
            print('\nException encountered:', e)
            traceback.print_exc()
