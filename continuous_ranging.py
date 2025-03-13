import os
import traceback
import time
from datetime import datetime
import queue
import random
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from thymiodirect import Thymio, Connection
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter
from music_v2 import music
from sonar import sonar

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)

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
    return padded_windowed_vec/max(padded_windowed_vec)

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

def windower(a):
    window = signal.windows.tukey(len(a), alpha=0.2)
    if len(a.shape) > 1:
        window = np.reshape(window, (-1, 1))
    windowed_a = a * window
    return windowed_a

if __name__ == "__main__":

    # save_recordings = True
    # rec_dir = './recordings/'
    # if save_recordings:
    #     print('\nRecordings will be saved in', rec_dir)
    #     if not os.path.exists(rec_dir):
    #         os.makedirs(rec_dir)
    
    fs = 192e3

    C_AIR = 343
    min_distance = 10e-2
    discarded_samples = int(np.floor((min_distance*2)/C_AIR*fs))

    method = 'music'
    if method == 'das':
        spatial_filter = das_filter
        doa_thresh = -55
    elif method == 'music':
        spatial_filter = music
        doa_thresh = -12
    dur = 3e-3
    hi_freq = 55e3
    low_freq = 20e3
    delta_freq = 7500
    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)
    sig = pow_two_pad_and_window(chirp, show=False)

    silence_dur = 20 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = pow_two(np.concatenate((sig, silence_vec)))
    output_sig = np.float32(np.reshape(full_sig, (-1, 1)))

    audio_in_data = queue.Queue()

    current_frame = 0
    def callback(indata, outdata, frames, time, status):
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

    device = get_soundcard_iostream(sd.query_devices())
    activate_mics()

    try:
        # real robot
        port = Connection.serial_default_port()
        try:
            speed = 300
            rot_speed = 150
            lateral_threshold = 1000
            ground_threshold = 10000
            air_threshold = 50
            output_threshold = -48 # [dB]
            distance_threshold = 30 # [cm]

            th = Thymio(serial_port=port,
            on_connect=lambda node_id: print(f'\nThymio {node_id} is connected'))
            th.connect()
            robot = th[th.first_node()]            
            # Delay to allow robot initialization of all variables
            time.sleep(1)
            
            robot['motor.left.target'] = speed
            robot['motor.right.target'] = speed
            
            while True:
                # Robot left the ground
                if (robot['prox.ground.reflected'][0] < air_threshold or robot['prox.ground.reflected'][1] < air_threshold):
                    print('Robot left the ground')
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
                
                stream = sd.Stream(samplerate=fs,
                        blocksize=0,
                        device=device,
                        channels=(8, 1),
                        callback=callback,
                        latency='low')
                
                with stream:
                    while stream.active:
                        pass
        
                current_frame = 0
                all_input_audio = []
                while not audio_in_data.empty():
                    all_input_audio.append(audio_in_data.get())
                input_audio = np.concatenate(all_input_audio)

                # if save_recordings:
                #     now = datetime.now()
                #     filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S-%f') + '.wav')
                #     sf.write(filename, input_audio, int(fs))

                dB_rms = 20*np.log10(np.mean(np.std(input_audio, axis=0))) # Battery is dead or not connected
                if dB_rms > output_threshold:

                    filtered_signals = signal.correlate(input_audio, np.reshape(sig, (-1, 1)), 'same', method='fft')
                    roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)

                    distance, direct_path = sonar(roll_filt_sigs, discarded_samples, fs)
                    distance = distance*100 # [m] to [cm]

                    if distance < distance_threshold and distance > 0:
                        print('Estimated distance: %3.1f' % distance, '[cm]')
                        robot['leds.bottom.left'] = [0, 255, 0]
                        robot['leds.bottom.right'] = [0, 255, 0]

                        # direction = random.choice(['l', 'r'])
                        # if direction == 'l':
                        #     robot['motor.left.target'] = -rot_speed
                        #     robot['motor.right.target'] = rot_speed
                        # else:
                        #     robot['motor.left.target'] = rot_speed
                        #     robot['motor.right.target'] = -rot_speed
                        # time.sleep(1)


                        theta, p = spatial_filter(
                                                    windower(roll_filt_sigs[direct_path + 40:direct_path + 40 + 380]), 
                                                    fs=fs, nch=roll_filt_sigs.shape[1], d=2.70e-3, 
                                                    bw=(low_freq + delta_freq, hi_freq - delta_freq)
                                                )
                        p_dB = 20*np.log10(p)
                        if max(p_dB) > doa_thresh:
                            robot['leds.bottom.left'] = [0, 255, 0]
                            robot['leds.bottom.right'] = [0, 255, 0]
                            doa_index = np.argmax(p_dB)
                            theta_hat = theta[doa_index]
                            print('\nEstimated DoA: %.2f [deg]' % theta_hat)
                            if theta_hat > 0:
                                robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 255, 255]
                                direction = 'r'
                            elif theta_hat < 0:
                                robot['leds.circle'] = [0, 255, 255, 0, 0, 0, 0, 0]
                                direction = 'l'
                            else:
                                robot['leds.circle'] = [255, 0, 0, 0, 0, 0, 0, 0]
                                direction = random.choice(['l', 'r'])

                            if direction == 'l':
                                robot['motor.left.target'] = -rot_speed
                                robot['motor.right.target'] = rot_speed
                            else:
                                robot['motor.left.target'] = rot_speed
                                robot['motor.right.target'] = -rot_speed
                            time.sleep(1)

                            robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            print('No DoA detected')
                        
                        robot['leds.bottom.left'] = [0, 0, 0]
                        robot['leds.bottom.right'] = [0, 0, 0]
                        robot['motor.left.target'] = speed
                        robot['motor.right.target'] = speed
                    
                else:
                    print('Low output level. Dead battery?')
                    
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
                elif robot['prox.horizontal'][4] > lateral_threshold:
                    robot['leds.bottom.right'] = [0, 0, 255]
                    robot['motor.left.target'] = -rot_speed
                    robot['motor.right.target'] = rot_speed
                    while robot['prox.horizontal'][4] > lateral_threshold:
                        pass
                    robot['leds.bottom.right'] = [0, 0, 0]
                    robot['motor.left.target'] = speed
                    robot['motor.right.target'] = speed
                        
        except KeyboardInterrupt:            
            print('Terminated by user')
        finally:
            try:
                robot['motor.left.target'] = 0
                robot['motor.right.target'] = 0
                robot['leds.bottom.left'] = 0
                robot['leds.bottom.right'] = 0
                robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 0, 0]
                th.disconnect()
            except Exception as e:
                print('Exception encountered:', e)
            finally:
                print('Fin')

    except IndexError:
        t_plot = np.linspace(0, input_audio.shape[0]/fs, input_audio.shape[0])
        fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
        plt.suptitle('Recorded Audio')
        for i in range(input_audio.shape[1]//2):
            for j in range(2):
                ax[i, j].plot(t_plot, input_audio[:, 2*i+j])
                ax[i, j].set_title('Channel %d' % (2*i+j+1))
                ax[i, j].minorticks_on()
                ax[i, j].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                ax[i, j].grid()
        plt.tight_layout()
        plt.show()
    except Exception:
        print(traceback.format_exc())
