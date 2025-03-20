from thymiodirect import Connection, Thymio
import time
import numpy as np

def angle_to_time(angle, speed):
    A = 612.33
    B = -0.94
    t = A*speed**B    
    return t * angle / 360

try:
    port = Connection.serial_default_port()
    th = Thymio(serial_port=port, 
                on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
    th.connect()
    robot_id = th.first_node()
    robot = th[robot_id]
    speed = 321

    angle = 360
    time_to_turn = angle_to_time(angle, speed)
    robot['motor.left.target'] = speed
    robot['motor.right.target'] = -speed
    now = time.time()
    while time.time() - now < time_to_turn:
        pass
    robot['motor.left.target'] = 0
    robot['motor.right.target'] = 0
    time.sleep(1)
    try:
        th.disconnect()
    except Exception as e:
            print(e)
except Exception as e:
    print(e)