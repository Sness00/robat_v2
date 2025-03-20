from thymiodirect import Connection, Thymio
import time
import numpy as np

try:
    port = Connection.serial_default_port()
    th = Thymio(serial_port=port, 
                on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
    th.connect()
    robot_id = th.first_node()
    robot = th[robot_id]
    speed = 350
    robot['motor.left.target'] = speed
    robot['motor.right.target'] = -speed
    
    now = time.time()
    while True:
            if robot['prox.horizontal'][4] > 600:
                print(time.time() - now)
                break
    robot['motor.left.target'] = 0
    robot['motor.right.target'] = 0
    time.sleep(1)
    try:
        th.disconnect()
    except Exception as e:
            print(e)
except Exception as e:
    print(e)