import os
import time
from datetime import datetime
import queue
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from thymiodirect import Thymio, Connection
from broadcast_pcmd3180 import activate_mics
from das_v2 import das_filter

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)

def pow_two_pad_and_window(vec, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.2)
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

def sonar(signals, output_sig, Fs=192e3):
    obst_distance = 0
    counter = 0
    for i in np.arange(signals.shape[1]):
        filtered_signal = signal.correlate(signals[:, i], output_sig, 'same', method='fft')
        smoothed_signal = np.abs(signal.hilbert(filtered_signal))
        peaks, _ = signal.find_peaks(smoothed_signal, prominence=12)
        if len(peaks) > 1:
            obst_distance += (peaks[1] - peaks[0])/Fs*343/2 + 0.025
            counter += 1
    if counter > 5:
        return obst_distance/counter
    else:
        return 0
    
def sonar_1ch(signals, output_sig, Fs=192e3):
    distances = [0, 0]
    filtered_signals = signal.correlate(signals, np.reshape(output_sig, (-1, 1)), 'same', method='fft')
    envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))
    c = 0
    for i in np.arange(1, envelopes.shape[1], 2):
        mean_env = np.mean(envelopes[:, i-1:i], axis=1)
        peaks, _ = signal.find_peaks(mean_env, prominence=7, distance=30)
        if len(peaks) > 1:
            obst_distance = (peaks[1] - peaks[0])/Fs*343/2 + 0.025
            distances[c] = obst_distance*100
        c += 1        
    return distances

def mean_env_sonar(signals, output_sig, Fs=192e3):
    filtered_signals = signal.correlate(signals, np.reshape(output_sig, (-1, 1)), 'same', method='fft')
    envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))
    mean_env = np.mean(envelopes, axis=1)

    peaks, _ = signal.find_peaks(mean_env, prominence=7, distance=30)
    if len(peaks) > 1:
        obst_distance = (peaks[1] - peaks[0])/Fs*343/2 + 0.025
        return obst_distance, filtered_signals, peaks[0]
    else:
        return 0, filtered_signals, None

def windower(a):
    window = signal.windows.tukey(len(a), alpha=0.2)
    if len(a.shape) > 1:
        window = np.reshape(window, (-1, 1))
    windowed_a = a * window
    return windowed_a

if __name__ == "__main__":

    save_recordings = True
    rec_dir = './recordings/'
    if save_recordings:
        print('\nRecordings will be saved in', rec_dir)
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)

    fs = 192e3
    dur = 2e-3
    hi_freq = 55e3
    low_freq = 25e3
    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)
    sig = pow_two_pad_and_window(chirp, show=False)

    silence_dur = 15 # [ms]
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
            speed = 0
            rot_speed = 150
            lateral_threshold = 1000
            ground_threshold = 10000
            air_threshold = 50
            output_threshold = -40
            distance_threshold = 30

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
                    while robot['prox.ground.reflected'][0] > ground_threshold:
                        robot['motor.left.target'] = rot_speed
                        robot['motor.right.target'] = -rot_speed
                    robot['leds.bottom.left'] = [0, 0, 0]
                    robot['leds.bottom.right'] = [0, 0, 0]
                    robot['motor.left.target'] = speed
                    robot['motor.right.target'] = speed
                # Right ground sensor
                elif robot['prox.ground.reflected'][1] > ground_threshold:
                    robot['leds.bottom.left'] = [255, 0, 0]
                    robot['leds.bottom.right'] = [255, 0, 0]
                    while robot['prox.ground.reflected'][1] > ground_threshold:
                        robot['motor.left.target'] = -rot_speed
                        robot['motor.right.target'] = rot_speed
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
                if save_recordings:
                    now = datetime.now()
                    filename = os.path.join(rec_dir, now.strftime('%Y%m%d_%H-%M-%S-%f') + '.wav')
                    sf.write(filename, input_audio, int(fs))
                db_rms = 20*np.log10(np.std(input_audio))
                # Battery is dead or not connected
                if db_rms < output_threshold:
                    print('Low output level. Dead battery?')
                else:
                    channels_12_78 = np.hstack(
                        (np.reshape(input_audio[:, 0], (-1, 1)),
                         np.reshape(input_audio[:, 1], (-1, 1)),
                         np.reshape(input_audio[:, 6], (-1, 1)),
                         np.reshape(input_audio[:, 7], (-1, 1)))
                            )
                    
                    dist_l, dist_r = sonar_1ch(channels_12_78, sig, Fs=fs)

                    distance, filt_sigs, direct_path = mean_env_sonar(input_audio, sig, Fs=fs)
                    distance = distance*100
                    if dist_l < distance:
                        distance = dist_l
                    if dist_r < distance:
                        distance = dist_r
                    if distance < distance_threshold and distance > 0:
                        print('Estimated distance: %3.1f' % distance, '[cm]')
                        print('Estimated distance ch1: %3.1f' % dist_l, '[cm]')
                        print('Estimated distance ch8: %3.1f' % dist_r, '[cm]')                    
                        theta2, p_das2 = das_filter(windower(filt_sigs[direct_path+96:direct_path+96+336]), fs=fs, nch=filt_sigs.shape[1], d=2.70e-3, bw=(low_freq, hi_freq))
                        if max(p_das2) > 0.005:
                            robot['leds.bottom.left'] = [0, 255, 0]
                            robot['leds.bottom.right'] = [0, 255, 0]
                            doa_index = np.argmax(p_das2)
                            theta_hat = theta2[doa_index]
                            print('\nEstimated DoA: %.2f [deg]\n' % theta_hat)
                            if theta_hat > 0:
                                robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 255, 255]
                            elif theta_hat < 0:
                                robot['leds.circle'] = [0, 255, 255, 0, 0, 0, 0, 0]
                            else:
                                robot['leds.circle'] = [255, 0, 0, 0, 0, 0, 0, 0]
                            current_time = time.time()
                            while(time.time() - current_time) < 1:
                                if theta_hat <= 0:
                                    robot['motor.left.target'] = -rot_speed
                                    robot['motor.right.target'] = rot_speed
                                else:
                                    robot['motor.left.target'] = rot_speed
                                    robot['motor.right.target'] = -rot_speed
                            robot['leds.circle'] = [0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            print('No DoA detected')
                        
                        robot['leds.bottom.left'] = [0, 0, 0]
                        robot['leds.bottom.right'] = [0, 0, 0]
                        robot['motor.left.target'] = speed
                        robot['motor.right.target'] = speed

                    #Left proximity sensor
                    if robot['prox.horizontal'][0] > lateral_threshold:
                        robot['leds.bottom.left'] = [0, 0, 255]
                        while robot['prox.horizontal'][0] > lateral_threshold:
                            robot['motor.left.target'] = rot_speed
                            robot['motor.right.target'] = -rot_speed
                        robot['leds.bottom.left'] = [0, 0, 0]
                        robot['motor.left.target'] = speed
                        robot['motor.right.target'] = speed
                    # Right proximity sensor
                    elif robot['prox.horizontal'][4] > lateral_threshold:
                        robot['leds.bottom.right'] = [0, 0, 255]
                        while robot['prox.horizontal'][4] > lateral_threshold:
                            robot['motor.left.target'] = -rot_speed
                            robot['motor.right.target'] = rot_speed
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
    except Exception as e:
        print(e)
